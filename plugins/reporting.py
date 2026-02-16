import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterstats import zonal_stats
from pathlib import Path
import tempfile
import os
from typing import Any, List, Dict, Optional

try:
    import jenkspy
    JENKSPY_AVAILABLE = True
except ImportError:
    JENKSPY_AVAILABLE = False

from murasa.core.normalize import classify_jenks
from murasa.core.analysis import calculate_zonal_stats

from murasa.core.data_source import DataRegistry, SourceType
from murasa.core.logger import logger
from murasa.core.base import PostProcessorPlugin

class VectorExplainabilityPlugin(PostProcessorPlugin):
    """
    Plugin to aggregate raster risk to vector (Kelurahan/Desa) 
    and provide explainable reasoning for the risk level.
    """

    def __init__(self, name: str = "vector_explainability"):
        super().__init__(name)

    def post_process(self, 
                     registry: DataRegistry, 
                     risk_raster: np.ndarray, 
                     transform: Any, 
                     crs: str, 
                     output_dir: Path,
                     config: Any = None):
        
        logger.info(f"Running Post-Processor: {self.name}")
        
        if not registry.has('admin'):
            logger.warning("   No admin boundaries found in registry. Skipping vector analysis.")
            return
            
        admin_source = registry.get('admin')
        gdf_admin = admin_source.data.to_crs(crs)
        
        weights = config.weights if config else None

        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp_risk_path = tmp.name
            
        try:
            with rasterio.open(
                tmp_risk_path, 'w',
                driver='GTiff',
                height=risk_raster.shape[0],
                width=risk_raster.shape[1],
                count=1,
                dtype=np.float32,
                crs=crs,
                transform=transform,
                nodata=np.nan
            ) as dst:
                dst.write(risk_raster, 1)

            classified_raster, _ = self._classify_raster_pixels(risk_raster)
            
            logger.info("   Computing zonal statistics (Categorical)...")
            
            stats = calculate_zonal_stats(
                gdf_admin,
                classified_raster,
                stats=['count', 'mean', 'max'],
                categorical=True,
                nodata=0,
                transform=transform,
                crs=crs
            )
            
            results = []
            for idx, (row, stat) in enumerate(zip(gdf_admin.itertuples(), stats)):
                if not stat or stat.get('count', 0) == 0:
                    continue
                    
                total_pixels = stat.get('count', 0)
                
                pcts = {}
                for cls in range(1, 7):
                    count = stat.get(cls, 0)
                    pcts[f"pct_class_{cls}"] = (count / total_pixels * 100) if total_pixels > 0 else 0
                
                pct_extreme = pcts.get("pct_class_6", 0)
                pct_very_high = pcts.get("pct_class_5", 0)
                pct_high = pcts.get("pct_class_4", 0)
                
                data = {
                    'geometry': row.geometry,
                    'area_ha': getattr(row, 'area_ha', 0),
                    'risk_score': stat.get('mean', 0),
                    'risk_max': stat.get('max', 0),
                    'pixel_count': total_pixels,
                    'pct_extreme': pct_extreme,
                    'pct_very_high': pct_very_high,
                    'pct_high': pct_high
                }
                
                for col in gdf_admin.columns:
                    if col not in data and col != 'geometry':
                        data[col] = getattr(row, col)
                        
                results.append(data)

            if not results:
                logger.warning("No results to export.")
                return

            out_gdf = gpd.GeoDataFrame(results, crs=crs)
            weights = config.weights if config else None
            self._enrich_factors(out_gdf, registry, transform, crs, weights)
            
            method = config.analysis.classification_method if config and hasattr(config, 'analysis') else 'jenks'
            out_gdf = self._classify_vector_legacy(out_gdf, method)
            
            out_gdf['reason'] = out_gdf.apply(self._generate_explanation, axis=1)
            
            self._add_explainability_metrics(out_gdf)
            

            out_shp = output_dir / "flood_risk_vector_explained.shp"
            out_csv = output_dir / "flood_risk_explainability.csv"
            
            try:
                out_gdf.to_file(out_shp)
                logger.info(f"   Saved explained vector output: {out_shp.name}")
            except Exception as e:
                logger.error(f"Failed to save SHP: {e}")
                
            df_export = pd.DataFrame(out_gdf.drop(columns='geometry'))
            df_export.to_csv(out_csv, index=False)
            logger.info(f"   Saved CSV output: {out_csv.name}")
            
            if config and hasattr(config, 'analysis') and config.analysis.enable_sensitivity_analysis:
                    self._run_sensitivity_check(out_gdf, config.analysis.sensitivity_scenarios)
                



            
        finally:
            if os.path.exists(tmp_risk_path):
                os.remove(tmp_risk_path)

    def _enrich_factors(self, gdf, registry, transform, crs, weights):
        """Extract mean values for each factor using risk layers"""
        logger.info("   Extracting contributing factors...")
        
        factor_map = {
            'rainfall': 'p_rainfall',
            'elevation': 'p_elevation',
            'slope': 'p_slope',
            'twi': 'p_twi',
            'land_use': 'p_landuse',
            'proximity': 'p_prox'
        }
        
        for key, col_name in factor_map.items():
            risk_key = f"risk_{key}"
            source = None
            
            if registry.has(risk_key):
                source = registry.get(risk_key)
            elif registry.has(key) and registry.get(key).source_type == SourceType.RASTER:
                 source = registry.get(key)
            
            if source is None:
                gdf[col_name] = 0.0
                continue
                
            data = source.data
            
            stats = calculate_zonal_stats(
                gdf,
                source,
                stats=['mean'],
                nodata=np.nan,
                transform=transform,
                crs=crs
            )
            
            gdf[col_name] = [s['mean'] if s and s['mean'] is not None else 0.0 for s in stats]
                    
        if weights:
            w_dict = weights.as_dict() if hasattr(weights, 'as_dict') else dict(weights)
            gdf['c_rainfall'] = gdf['p_rainfall'] * w_dict.get('rainfall', 0)
            gdf['c_elev'] = gdf['p_elevation'] * w_dict.get('elevation', 0)
            gdf['c_slope'] = gdf['p_slope'] * w_dict.get('slope', 0)
            gdf['c_twi'] = gdf['p_twi'] * w_dict.get('twi', 0)
            gdf['c_landuse'] = gdf['p_landuse'] * w_dict.get('land_use', 0)
            gdf['c_prox'] = gdf['p_prox'] * w_dict.get('proximity', 0)

    def _classify_raster_pixels(self, risk_raster):
        """Classify 0-1 float raster into 1-6 integer raster using Jenks"""
        logger.info("   Classifying Raster Pixels (1-6 Scale)...")
        return classify_jenks(risk_raster, n_classes=6)

    def _classify_vector_legacy(self, gdf, method):
        """Classify vector based on composite score (Legacy Logic)"""
        gdf['composite_risk'] = (
            gdf['pct_extreme'] * 2.0 + 
            gdf['pct_very_high'] * 1.5 +
            gdf['pct_high'] * 1.0 + 
            gdf['risk_score'] * 0.5  
        )
        
        labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Extreme']
        n_classes = len(labels)
        
        if method == 'jenks' and JENKSPY_AVAILABLE:
            try:
                breaks = jenkspy.jenks_breaks(gdf['composite_risk'], n_classes=n_classes)
                gdf['severity'] = pd.cut(gdf['composite_risk'], bins=breaks, labels=labels, include_lowest=True)
            except Exception as e:
                logger.warning(f"Jenks on vector failed: {e}")
                try:
                    gdf['severity'] = pd.qcut(gdf['composite_risk'], q=n_classes, labels=labels, duplicates='drop')
                except ValueError:
                    gdf['severity'] = 'Low' 
        else:
            try:
                gdf['severity'] = pd.qcut(gdf['composite_risk'], q=n_classes, labels=labels, duplicates='drop')
            except ValueError:
                gdf['severity'] = 'Low'
            
        return gdf

    def _classify_vector_expert(self, gdf):
        """
        Original rule-based classification (Explicit Thresholds).
        """
        logger.info("   Applying thresholds (Rule-Based)...")
        
        severity_list = []
        
        for _, row in gdf.iterrows():
            extreme_pct = row.get('pct_extreme', 0)
            very_high_pct = row.get('pct_very_high', 0)
            high_pct = row.get('pct_high', 0)
            risk_score = row.get('risk_score', 0)
            
            if extreme_pct > 35 or (extreme_pct > 25 and risk_score >= 5.5):
                severity = 'Extreme'
            elif very_high_pct > 30 or extreme_pct > 15 or risk_score >= 5.0:
                severity = 'Very High'
            elif high_pct > 30 or very_high_pct > 20 or risk_score >= 4.2:
                severity = 'High'
            elif risk_score >= 3.5:
                severity = 'Medium'
            elif risk_score >= 2.5:
                severity = 'Low'
            else:
                severity = 'Very Low'
                
            severity_list.append(severity)
            
        gdf['severity'] = severity_list
        return gdf

    def _classify_risk(self, gdf, method='expert'):
        """Classify vector risk scores"""
        logger.info(f"   Classifying risk using: {method}")
        
        if method == 'expert':
            return self._classify_vector_expert(gdf)
        elif method == 'jenks':
            return self._classify_vector_legacy(gdf, 'jenks') 
        else:
            return self._classify_vector_legacy(gdf, 'quantile')


    def _add_explainability_metrics(self, gdf):
        """Add dominant factor and confidence"""
        factors = ['c_rainfall', 'c_elev', 'c_slope', 'c_twi', 'c_landuse', 'c_prox']
        names = ['Rainfall', 'Elevation', 'Slope', 'TWI', 'Land Use', 'Proximity']
        
        dominant = []
        confidence = []
        certainty = []
        
        for idx, row in gdf.iterrows():
            vals = [row.get(f, 0) for f in factors]
            total = sum(vals)
            
            if total == 0:
                dominant.append("None")
                confidence.append(0)
                certainty.append(0)
                continue
                
            max_idx = np.argmax(vals)
            dom_name = names[max_idx]
            conf = vals[max_idx] / total 
            
            norm_vals = np.array(vals) / total
            var = np.var(norm_vals)
            cert = np.sqrt(var) * 2.5 
            cert = min(cert, 1.0)
            
            dominant.append(dom_name)
            confidence.append(round(conf, 3))
            certainty.append(round(cert, 3))
            
        gdf['dominant'] = dominant
        gdf['conf'] = confidence
        gdf['cert'] = certainty

    def _generate_explanation(self, row):
        """Generate human-readable explanation"""
        reasons = []
        reasons.append(f"Risk Level: {row.get('severity', 'Unknown')} ({row['risk_score']:.2f})")
        
        factors = []
        
        if row.get('p_rainfall', 0) > 0.7: factors.append(f"High Rainfall ({row['p_rainfall']:.2f})")
        if row.get('p_elevation', 0) > 0.7: factors.append(f"Low Elevation ({row['p_elevation']:.2f})")
        if row.get('p_slope', 0) > 0.7: factors.append(f"Flat Terrain ({row['p_slope']:.2f})")
        if row.get('p_twi', 0) > 0.7: factors.append(f"Water Accumulation ({row['p_twi']:.2f})")
        if row.get('p_landuse', 0) > 0.7: factors.append(f"Impervious/Vuln Land ({row['p_landuse']:.2f})")
        if row.get('p_prox', 0) > 0.7: factors.append(f"Near River ({row['p_prox']:.2f})")
        
        if factors:
            reasons.append("Driven by: " + ", ".join(factors))
        else:
            dom = row.get('dominant', 'None')
            if dom != 'None':
                reasons.append(f"Mainly driven by {dom}")
            
        return " | ".join(reasons)

    def _run_sensitivity_check(self, gdf, scenarios):
        """Check against sensitivity thresholds"""
        logger.info("   Running Sensitivity Check...")
        for name, thresh in scenarios.items():
            extreme_count = len(gdf[gdf['risk_score'] >= thresh.get('extreme_score', 0.8)])
            logger.info(f"      Scenario {name}: {extreme_count} Extreme Risk Units (> {thresh.get('extreme_score')})")
