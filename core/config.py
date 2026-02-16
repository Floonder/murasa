from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import json
from .logger import logger, log_success, log_error


@dataclass
class FactorConfig:
    name: str
    weight: float = 0.0
    source_path: Optional[Path] = None
    source_type: str = "raster"
    processor: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.source_path, str):
            self.source_path = Path(self.source_path)


@dataclass
class SpatialConfig:
    target_crs: str = "EPSG:4326"
    metric_crs: str = "EPSG:3857"
    resolution: float = 10.0
    study_area_key: str = "admin"

    filter_province: Optional[List[str]] = None
    filter_city: Optional[List[str]] = None
    filter_district: Optional[List[str]] = None

    def get_filter_level(self) -> str:
        if self.filter_district:
            return "district"
        elif self.filter_city:
            return "city"
        elif self.filter_province:
            return "province"
        return "full"

    def get_active_filter(self) -> Optional[List[str]]:
        if self.filter_district:
            return self.filter_district
        elif self.filter_city:
            return self.filter_city
        elif self.filter_province:
            return self.filter_province
        return None


@dataclass
class OutputConfig:
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    formats: List[str] = field(default_factory=lambda: ["geojson"])
    generate_report: bool = True

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


@dataclass
class ClassificationConfig:
    method: str = "quantile"
    num_classes: int = 5
    class_names: List[str] = field(default_factory=lambda: [
        "Very Low", "Low", "Moderate", "High", "Very High"
    ])
    class_colors: List[str] = field(default_factory=lambda: [
    ])

    thresholds: Optional[List[float]] = None


@dataclass
class PathsConfig:
    dem: Optional[Path] = None
    slope: Optional[Path] = None
    twi: Optional[Path] = None
    rainfall: Optional[Path] = None
    river: Optional[Path] = None
    admin_dirs: List[Path] = field(default_factory=list)
    output_dir: Path = field(default_factory=lambda: Path("./output"))

    def __post_init__(self):
        for attr in ('dem', 'slope', 'twi', 'rainfall', 'river'):
            val = getattr(self, attr)
            if isinstance(val, str):
                setattr(self, attr, Path(val))
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        self.admin_dirs = [Path(p) if isinstance(p, str) else p for p in self.admin_dirs]


@dataclass
class WeightsConfig:
    rainfall: float = 0.0
    elevation: float = 0.0
    slope: float = 0.0
    twi: float = 0.0
    proximity: float = 0.0
    land_use: float = 0.0

    def total(self) -> float:
        return (self.rainfall + self.elevation + self.slope
                + self.twi + self.proximity + self.land_use)

    def as_dict(self) -> Dict[str, float]:
        return {
            'rainfall': self.rainfall,
            'elevation': self.elevation,
            'slope': self.slope,
            'twi': self.twi,
            'proximity': self.proximity,
            'land_use': self.land_use,
        }


@dataclass
class EngineConfig:
    name: str = "unnamed_analysis"
    description: str = ""
    analysis_type: str = "susceptibility"

    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)

    factors: Dict[str, FactorConfig] = field(default_factory=dict)

    admin_dirs: List[Path] = field(default_factory=list)
    admin_filename: str = "ADMINISTRASIDESA_AR_25K.shp"

    parameters: Dict[str, Any] = field(default_factory=dict)

    paths: PathsConfig = field(default_factory=PathsConfig)
    weights: WeightsConfig = field(default_factory=WeightsConfig)
    plugin_parameters: Dict[str, Any] = field(default_factory=dict)

    def get_weights(self) -> Dict[str, float]:
        return {name: f.weight for name, f in self.factors.items()}

    def get_factor_path(self, factor_name: str) -> Optional[Path]:
        if factor_name in self.factors:
            return self.factors[factor_name].source_path
        return None

    def validate_weights(self) -> None:
        if not self.factors:
            return
        total = sum(f.weight for f in self.factors.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total:.3f}")

    def normalize_weights(self) -> None:
        total = sum(f.weight for f in self.factors.values())
        if total > 0:
            for factor in self.factors.values():
                factor.weight /= total
            log_success(f"Weights normalized (total was {total:.3f})")

    def validate(self) -> None:
        self.validate_weights()
        self.output.output_dir.mkdir(parents=True, exist_ok=True)

        missing = []
        for name, factor in self.factors.items():
            if factor.source_path and not factor.source_path.exists():
                missing.append(f"{name}: {factor.source_path}")

        if missing:
            for m in missing:
                log_error(f"Factor path not found: {m}")

        log_success("Configuration validated")

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'EngineConfig':
        logger.info(f"Loading config from: {yaml_path}")
        yaml_path = Path(yaml_path)
        base_dir = yaml_path.parent.resolve()

        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to read YAML: {e}")
            raise

        project = data.get('project', {})
        config = cls(
            name=project.get('name', data.get('name', 'unnamed')),
            description=project.get('description', data.get('description', '')),
            analysis_type=data.get('analysis_type', 'susceptibility'),
        )

        if 'spatial' in data:
            config.spatial = SpatialConfig(**data['spatial'])

        if 'output' in data:
            out_data = data['output']
            config.output = OutputConfig(
                output_dir=Path(out_data.get('dir', './output')),
                formats=out_data.get('formats', ['geojson']),
                generate_report=out_data.get('generate_report', True)
            )

        if 'classification' in data:
            config.classification = ClassificationConfig(**data['classification'])

        is_migrated = 'paths' in data and 'weights' in data

        if is_migrated:
            config._load_migrated_format(data, base_dir)
        else:
            config._load_factors_format(data, base_dir)

        if 'admin_dirs' in data:
            config.admin_dirs = [
                Path(p) if Path(p).is_absolute() else base_dir / p
                for p in data['admin_dirs']
            ]

        if 'admin_filename' in data:
            config.admin_filename = data['admin_filename']

        config.parameters = data.get('parameters', {})

        config.validate()
        return config

    def _load_migrated_format(self, data: dict, base_dir: Path) -> None:
        paths_data = data.get('paths', {})

        resolve = lambda p: Path(p).resolve() if Path(p).is_absolute() else (base_dir / p).resolve()

        self.paths = PathsConfig(
            dem=resolve(paths_data['dem']) if 'dem' in paths_data else None,
            slope=resolve(paths_data['slope']) if 'slope' in paths_data else None,
            twi=resolve(paths_data['twi']) if 'twi' in paths_data else None,
            rainfall=resolve(paths_data['rainfall']) if 'rainfall' in paths_data else None,
            river=resolve(paths_data['river']) if 'river' in paths_data else None,
            admin_dirs=[
                resolve(p) for p in paths_data.get('admin_dirs', [])
            ],
            output_dir=resolve(paths_data.get('output_dir', './output'))
        )

        self.admin_dirs = self.paths.admin_dirs
        self.output = OutputConfig(output_dir=self.paths.output_dir)

        weights_data = data.get('weights', {})
        self.weights = WeightsConfig(**weights_data)

        self.plugin_parameters = data.get('parameters', {})

        path_map = {
            'rainfall': self.paths.rainfall,
            'elevation': self.paths.dem,
            'slope': self.paths.slope,
            'twi': self.paths.twi,
        }
        source_type_map = {
            'rainfall': 'vector',
            'elevation': 'raster',
            'slope': 'raster',
            'twi': 'raster',
        }

        for factor_name, weight in self.weights.as_dict().items():
            if weight <= 0:
                continue
            source_path = path_map.get(factor_name)
            self.factors[factor_name] = FactorConfig(
                name=factor_name,
                weight=weight,
                source_path=source_path,
                source_type=source_type_map.get(factor_name, 'derived'),
                parameters=self.plugin_parameters.get(factor_name, {})
            )

        logger.info("  Config format: migrated (paths + weights)")

    def _load_factors_format(self, data: dict, base_dir: Path) -> None:
        for name, fdata in data.get('factors', {}).items():
            source_path = None
            if 'path' in fdata:
                p = Path(fdata['path'])
                source_path = p if p.is_absolute() else base_dir / p

            self.factors[name] = FactorConfig(
                name=name,
                weight=fdata.get('weight', 0.0),
                source_path=source_path,
                source_type=fdata.get('type', 'raster'),
                processor=fdata.get('processor'),
                parameters=fdata.get('parameters', {})
            )

        logger.info("  Config format: (factors)")

    def to_yaml(self, yaml_path: Path) -> None:
        data = {
            'name': self.name,
            'description': self.description,
            'analysis_type': self.analysis_type,
            'spatial': {
                'target_crs': self.spatial.target_crs,
                'metric_crs': self.spatial.metric_crs,
                'resolution': self.spatial.resolution,
                'filter_province': self.spatial.filter_province,
                'filter_city': self.spatial.filter_city,
                'filter_district': self.spatial.filter_district,
            },
            'output': {
                'dir': str(self.output.output_dir),
                'formats': self.output.formats,
                'generate_report': self.output.generate_report,
            },
            'classification': {
                'method': self.classification.method,
                'num_classes': self.classification.num_classes,
                'class_names': self.classification.class_names,
                'class_colors': self.classification.class_colors,
                'thresholds': self.classification.thresholds,
            },
            'factors': {
                name: {
                    'weight': f.weight,
                    'path': str(f.source_path) if f.source_path else None,
                    'type': f.source_type,
                    'processor': f.processor,
                    'parameters': f.parameters,
                }
                for name, f in self.factors.items()
            },
            'admin_dirs': [str(p) for p in self.admin_dirs],
            'admin_filename': self.admin_filename,
            'parameters': self.parameters,
        }

        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        log_success(f"Configuration saved to {yaml_path}")

    def to_json(self, json_path: Path) -> None:
        data = {
            'name': self.name,
            'description': self.description,
            'analysis_type': self.analysis_type,
            'factors': {
                name: {
                    'weight': f.weight,
                    'path': str(f.source_path) if f.source_path else None,
                    'type': f.source_type,
                    'processor': f.processor,
                    'parameters': f.parameters,
                }
                for name, f in self.factors.items()
            },
            'parameters': self.parameters,
        }

        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

        log_success(f"Configuration saved to {json_path}")
