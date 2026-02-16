import sys
import argparse
from pathlib import Path

from murasa.core.config import EngineConfig
from murasa.engine.risk_engine import RiskEngine
from murasa.core.data_source import DataSource, SourceType
from murasa.core.base import LoaderRegistry
from murasa.plugins import (
    RainfallPlugin, ElevationPlugin, SlopePlugin,
    TWIPlugin, ProximityPlugin, LandUsePlugin, WaterExclusionPlugin
)
from murasa.plugins.reporting import VectorExplainabilityPlugin
from murasa.core.logger import logger, log_section, log_success, log_error

PLUGIN_MAP = {
    'rainfall': lambda weight, params: RainfallPlugin(
        weight=weight,
        column_name=params.get('column_name', params.get('column', 'gridcode')),
        value_mapping=params.get('value_mapping')
    ),
    'elevation': lambda weight, params: ElevationPlugin(
        weight=weight,
        inverse=params.get('invert', True)
    ),
    'slope': lambda weight, params: SlopePlugin(
        weight=weight,
        inverse=params.get('invert', True)
    ),
    'twi': lambda weight, params: TWIPlugin(
        weight=weight
    ),
    'land_use': lambda weight, params: LandUsePlugin(
        weight=weight,
        coefficients=params.get('coefficients'),
        priorities=params.get('priorities')
    ),
    'proximity': lambda weight, params: ProximityPlugin(
        weight=weight,
        feature_name=params.get('feature', 'river'),
        max_distance=params.get('max_distance', 500)
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Flood Susceptibility Analysis")
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("flood_bandung_config.yaml"),
        help="Path to YAML config file"
    )
    return parser.parse_args()


def load_config(config_path: Path) -> EngineConfig:
    log_section("Loading configuration")

    if not config_path.exists():
        log_error(f"Config not found: {config_path}")
        sys.exit(1)

    config = EngineConfig.from_yaml(config_path)

    logger.info(f"  Analysis : {config.name}")
    logger.info(f"  CRS      : {config.spatial.target_crs}")
    logger.info(f"  Resolution: {config.spatial.resolution}m")

    if hasattr(config, 'factors') and config.factors:
        logger.info(f"  Factors  : {list(config.factors.keys())}")
    if hasattr(config, 'weights') and config.weights:
        logger.info(f"  Weights  : {config.weights}")

    return config


def init_engine(config: EngineConfig) -> RiskEngine:
    log_section("Init engine")

    engine = RiskEngine(config)
    engine.auto_register_from_config()

    return engine


def load_data(config: EngineConfig, engine: RiskEngine) -> None:
    log_section("Loading data")

    loader_registry = LoaderRegistry()
    loader_registry.auto_discover(Path(__file__).parent.parent / "plugins")
    loader_registry.load_all(config, engine.registry)


def register_parameters(config: EngineConfig, engine: RiskEngine) -> None:
    log_section("Registering risk parameters")

    registered = 0

    if hasattr(config, 'weights') and config.weights:
        weights = config.weights
        params_cfg = getattr(config, 'plugin_parameters',
                             getattr(config, 'parameters', {}))

        for factor_name, factory in PLUGIN_MAP.items():
            weight = getattr(weights, factor_name, None)
            if weight is None or weight <= 0:
                continue

            factor_params = params_cfg.get(factor_name, {})
            plugin = factory(weight, factor_params)
            if engine.register_parameter(plugin):
                registered += 1

    elif hasattr(config, 'factors') and config.factors:
        for factor_name, factor_cfg in config.factors.items():
            if factor_name not in PLUGIN_MAP:
                continue

            params = factor_cfg.parameters if hasattr(factor_cfg, 'parameters') else {}
            plugin = PLUGIN_MAP[factor_name](factor_cfg.weight, params)
            if engine.register_parameter(plugin):
                registered += 1

    engine.register_parameter(WaterExclusionPlugin())
    registered += 1

    engine.register_post_processor(VectorExplainabilityPlugin())

    logger.info(f"  Total: {registered} parameter(s) + 1 post-processor registered")


def run_analysis(engine: RiskEngine) -> None:
    log_section("Run complete analysis")

    engine.setup_grid()
    engine.set_study_area()

    risk_raster = engine.calculate_risk()

    engine.export_all()

    log_success(f"Results saved to: {engine.config.output.output_dir.resolve()}")


def main():
    args = parse_args()

    log_section("Flood susceptibility pipeline")
    logger.info(f"Config: {args.config}")

    config = load_config(args.config)
    engine = init_engine(config)
    load_data(config, engine)
    register_parameters(config, engine)

    engine.print_summary()

    try:
        run_analysis(engine)
        log_section("Completed")
    except Exception as e:
        log_error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
