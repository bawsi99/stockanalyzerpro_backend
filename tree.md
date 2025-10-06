# Backend Directory Tree

```
backend/
├── __init__.py
├── README.md
├── analysis/
│   ├── __pycache__/
│   ├── advanced_analysis.py
│   ├── datasets.py
│   ├── heavy_load.py
│   ├── inference.py
│   └── orchestrator.py
├── analysis_datasets/
├── analysis_service.log
├── api/
│   ├── __init__.py
│   ├── __pycache__/
│   ├── endpoints.py
│   ├── main.py
│   └── responses.py
├── backtesting/
│   └── backtesting.py
├── cache/
│   ├── [30 CSV files - cached data]
│   └── cache_metadata.json
├── config/
│   ├── __init__.py
│   ├── __pycache__/
│   ├── constraints.txt
│   ├── deployment_config.py
│   ├── requirements_data_service.txt
│   ├── requirements_database_service.txt
│   ├── requirements-dev.txt
│   ├── requirements.txt
│   ├── runtime.txt
│   └── storage_config.py
├── core/
│   ├── __init__.py
│   ├── __pycache__/
│   ├── chart_manager.py
│   ├── config.py
│   ├── instrument_filter.py
│   ├── redis_cache_manager.py
│   ├── redis_unified_cache_manager.py
│   ├── sector_manager.py
│   ├── supabase_client.py
│   └── utils.py
├── data/
│   ├── __init__.py
│   ├── __pycache__/
│   ├── analysis_datasets/
│   │   ├── analysis_datasets.json
│   │   ├── portfolio_dataset.json
│   │   ├── sector_dataset.json
│   │   └── trading_dataset.json
│   ├── cache/
│   │   ├── [279 CSV files - cached data]
│   │   └── cache_metadata.json
│   ├── enhanced_sector_data/
│   │   ├── filtered_equity_stocks.csv
│   │   ├── instrument_breakdown.json
│   │   ├── major_stocks.json
│   │   └── sector_performance.json
│   ├── sector_category/
│   │   ├── __init__.py
│   │   ├── auto.json
│   │   ├── banking.json
│   │   ├── consumer_durables.json
│   │   ├── consumption.json
│   │   ├── energy.json
│   │   ├── fmcg.json
│   │   ├── healthcare.json
│   │   ├── infrastructure.json
│   │   ├── it.json
│   │   ├── media.json
│   │   ├── metal.json
│   │   ├── oil_gas.json
│   │   ├── pharma.json
│   │   ├── realty.json
│   │   ├── telecom.json
│   │   └── transport.json
│   ├── signals/
│   │   ├── __init__.py
│   │   ├── __pycache__/
│   │   ├── config.py
│   │   ├── regimes.py
│   │   ├── schema.py
│   │   ├── scoring.py
│   │   └── weights_config.json
│   └── zerodha_instruments.csv
├── data_service.log
├── database_service.log
├── enhanced_sector_data/
├── gemini/
│   ├── __pycache__/
│   ├── context_engineer.py
│   ├── debug_config.py
│   ├── debug_logger.py
│   ├── error_utils.py
│   ├── gemini_client.py
│   ├── gemini_core.py
│   ├── image_utils.py
│   ├── parallel_pattern_detection.py
│   ├── prompt_manager.py
│   ├── schema.py
│   └── token_tracker.py
├── logs/
├── ml/
│   ├── __init__.py
│   ├── __pycache__/
│   ├── analysis/
│   │   ├── __pycache__/
│   │   ├── market_regime.py
│   │   ├── mtf_analysis.py
│   │   └── mtf_utils.py
│   ├── indicators/
│   │   ├── __pycache__/
│   │   ├── technical_indicators.py
│   │   └── volume_profile.py
│   ├── quant_system/
│   │   ├── __init__.py
│   │   ├── __pycache__/
│   │   ├── advanced/
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__/
│   │   │   ├── integration/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── __pycache__/
│   │   │   │   ├── phase2_manager.py
│   │   │   │   └── real_time_integrator.py
│   │   │   ├── models/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── __pycache__/
│   │   │   │   ├── ensemble_manager.py
│   │   │   │   ├── multimodal_fusion.py
│   │   │   │   ├── nbeats.py
│   │   │   │   └── tft.py
│   │   │   └── training/
│   │   │       ├── __init__.py
│   │   │       ├── __pycache__/
│   │   │       ├── advanced_strategies.py
│   │   │       ├── meta_learning.py
│   │   │       └── neural_architecture_search.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__/
│   │   │   ├── base_models.py
│   │   │   ├── config.py
│   │   │   ├── registry.py
│   │   │   └── utils.py
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__/
│   │   │   ├── dataset_builder.py
│   │   │   ├── datasets/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── cache/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── catboost_info/
│   │   │   │   │   │   ├── catboost_training.json
│   │   │   │   │   │   ├── learn/
│   │   │   │   │   │   │   └── events.out.tfevents
│   │   │   │   │   │   ├── learn_error.tsv
│   │   │   │   │   │   ├── time_left.tsv
│   │   │   │   │   │   └── tmp/
│   │   │   │   │   └── nas_results/
│   │   │   │   │       ├── nas_results_1755854300.json
│   │   │   │   │       └── nas_results_1755854599.json
│   │   │   │   ├── models/
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── enhanced_pattern_model.joblib
│   │   │   │   │   ├── pattern_catboost.joblib
│   │   │   │   │   └── pattern_registry.json
│   │   │   │   └── patterns/
│   │   │   │       ├── __init__.py
│   │   │   │       ├── robust_patterns_meta.json
│   │   │   │       ├── robust_patterns.parquet
│   │   │   │       ├── test_general_patterns_meta.json
│   │   │   │       └── test_general_patterns.parquet
│   │   │   ├── enhanced_pipeline.py
│   │   │   ├── market_data_integration.py
│   │   │   └── pipeline.py
│   │   ├── engines/
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__/
│   │   │   ├── hybrid_ml.py
│   │   │   ├── pattern_ml.py
│   │   │   ├── raw_data_ml.py
│   │   │   └── unified_manager.py
│   │   ├── evaluation/
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__/
│   │   │   ├── model_comparison.py
│   │   │   ├── price_analysis.py
│   │   │   └── robust_evaluation.py
│   │   ├── features/
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__/
│   │   │   ├── enhanced_features.py
│   │   │   ├── feature_engineer.py
│   │   │   └── technical_indicators.py
│   │   ├── REORGANIZATION_SUMMARY.md
│   │   ├── scripts/
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__/
│   │   │   ├── run_tests.py
│   │   │   └── train_models.py
│   │   ├── trading/
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__/
│   │   │   ├── backtesting/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── __pycache__/
│   │   │   │   └── engine.py
│   │   │   ├── execution/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── __pycache__/
│   │   │   │   ├── live_deployment.py
│   │   │   │   ├── production_system.py
│   │   │   │   └── simplified_system.py
│   │   │   └── strategies/
│   │   │       ├── __init__.py
│   │   │       ├── __pycache__/
│   │   │       ├── advanced_strategies.py
│   │   │       └── risk_management.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── __pycache__/
│   └── sector/
│       ├── __pycache__/
│       ├── benchmarking.py
│       ├── classifier.py
│       └── enhanced_classifier.py
├── myenv/
│   ├── bin/
│   │   ├── [Python virtual environment executables]
│   │   ├── activate
│   │   ├── activate.csh
│   │   ├── activate.fish
│   │   ├── Activate.ps1
│   │   ├── alembic
│   │   ├── automat-visualize
│   │   ├── build_sync
│   │   ├── cftp
│   │   ├── ckeygen
│   │   ├── conch
│   │   ├── convert-caffe2-to-onnx
│   │   ├── convert-onnx-to-caffe2
│   │   ├── dotenv
│   │   ├── f2py
│   │   ├── fastapi
│   │   ├── fonttools
│   │   ├── httpx
│   │   ├── isympy
│   │   ├── mailmail
│   │   ├── mako-render
│   │   ├── normalizer
│   │   ├── numpy-config
│   │   ├── optuna
│   │   ├── pip
│   │   ├── pip3
│   │   ├── pip3.10
│   │   ├── plotly_get_chrome
│   │   ├── pyftmerge
│   │   ├── pyftsubset
│   │   ├── pyhtmlizer
│   │   ├── pyrsa-decrypt
│   │   ├── pyrsa-encrypt
│   │   ├── pyrsa-keygen
│   │   ├── pyrsa-priv2pub
│   │   ├── pyrsa-sign
│   │   ├── pyrsa-verify
│   │   ├── python
│   │   ├── python3
│   │   ├── python3.10
│   │   ├── tests
│   │   ├── tkconch
│   │   ├── torchfrtrace
│   │   ├── torchrun
│   │   ├── trial
│   │   ├── ttx
│   │   ├── twist
│   │   ├── twistd
│   │   ├── uvicorn
│   │   └── wamp
│   ├── etc/
│   │   └── jupyter/
│   │       └── nbconfig/
│   │           └── notebook.d/
│   │               └── catboost-widget.json
│   ├── include/
│   ├── lib/
│   │   └── python3.10/
│   │       └── site-packages/
│   │           └── [Python packages]
│   ├── pyvenv.cfg
│   └── share/
│       ├── jupyter/
│       │   ├── labextensions/
│       │   │   ├── catboost-widget/
│       │   │   │   ├── package.json
│       │   │   │   └── static/
│       │   │   │       ├── 138.c8bd59d1ac66cac18152.js
│       │   │   │       ├── 479.d43617224b25ecf15b1d.js
│       │   │   │       ├── 479.d43617224b25ecf15b1d.js.LICENSE.txt
│       │   │   │       ├── 486.bafd26b008c3405f7750.js
│       │   │   │       ├── 486.bafd26b008c3405f7750.js.LICENSE.txt
│       │   │   │       ├── 755.297bcad6e07632169fc2.js
│       │   │   │       ├── 755.297bcad6e07632169fc2.js.LICENSE.txt
│       │   │   │       ├── 908.81f6af6c7d6425b98663.js
│       │   │   │       ├── remoteEntry.bb0af8065c14acd4d841.js
│       │   │   │       └── style.js
│       │   │   └── jupyterlab-plotly/
│       │   │       ├── install.json
│       │   │       ├── package.json
│       │   │       └── static/
│       │   │           ├── 340.2a23c8275d47a2531dae.js
│       │   │           ├── remoteEntry.5153b2c003c011c482e3.js
│       │   │           └── style.js
│       │   └── nbextensions/
│       │       └── catboost-widget/
│       │           ├── extension.js
│       │           └── index.js
│       └── man/
│           └── man1/
│               ├── isympy.1
│               └── ttx.1
├── output/
│   └── charts/
├── patterns/
│   ├── __pycache__/
│   ├── confirmation.py
│   ├── database.py
│   ├── recognition.py
│   └── visualization.py
├── prompts/
│   ├── final_stock_decision.txt
│   ├── image_analysis_comprehensive_overview.txt
│   ├── image_analysis_continuation_levels.txt
│   ├── image_analysis_reversal_patterns.txt
│   ├── image_analysis_volume_comprehensive.txt
│   ├── indicators_to_summary_and_json.txt
│   ├── meta_prompt.txt
│   ├── optimized_continuation_levels.txt
│   ├── optimized_final_decision.txt
│   ├── optimized_indicators_summary.txt
│   ├── optimized_mtf_comparison.txt
│   ├── optimized_pattern_analysis.txt
│   ├── optimized_reversal_patterns.txt
│   ├── optimized_technical_overview.txt
│   └── optimized_volume_analysis.txt
├── agents/
│   ├── __init__.py
│   ├── final_decision/
│   │   └── processor.py
│   ├── mtf_analysis/
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   └── processor.py
│   │   ├── integration_manager.py
│   │   ├── mtf_agents.py
│   │   └── orchestrator.py
│   ├── risk_analysis/
│   │   ├── __init__.py
│   │   ├── quantitative_risk/
│   │   │   └── processor.py
│   │   └── risk_llm_agent.py
│   ├── sector/
│   │   ├── benchmarking.py
│   │   ├── cache_manager.py
│   │   ├── classifier.py
│   │   └── enhanced_classifier.py
│   └── volume/
│       ├── __init__.py
│       └── volume_agents.py
├── risk/
│   ├── bayesian_scorer.py
│   └── scoring.py
├── scripts/
│   ├── __init__.py
│   ├── calibrate_all.py
│   ├── calibrate_weights.py
│   ├── generate_fixtures.py
│   ├── monitor_subscriptions.py
│   ├── run_production_services.py
│   ├── run_services.py
│   └── start_consolidated_service.py
├── services/
│   ├── __init__.py
│   ├── __pycache__/
│   ├── analysis_service.py
│   ├── central_data_provider.py
│   ├── consolidated_service.py
│   ├── data_service.py
│   ├── database_service.py
│   ├── enhanced_data_service.py
│   └── websocket_service.py
├── tree.md
├── utils/
│   └── memory_analyzer.py
├── zerodha/
│   ├── __init__.py
│   ├── __pycache__/
│   ├── cache/
│   │   ├── b69a63e156b3e23d8f3129be5d8028f1.csv
│   │   └── cache_metadata.json
│   ├── client.py
│   ├── token_updater.py
│   └── ws_client.py
└── zerodha_instruments.csv
```

## Summary

The backend directory contains a comprehensive stock analysis and trading system with the following main components:

### Core Modules
- **analysis/**: Advanced analysis modules including heavy load processing and orchestration
- **agents/**: Agent-based systems (volume, multi-timeframe analysis, risk analysis, sector)
- **api/**: FastAPI endpoints and responses
- **backtesting/**: Backtesting functionality for trading strategies
- **cache/**: Cached data storage (30 CSV files)
- **config/**: Configuration files and requirements
- **core/**: Core utilities, cache management, and client connections
- **data/**: Data storage, analysis datasets, and sector categorization
- **gemini/**: AI/ML integration with Google Gemini
- **ml/**: Machine learning models, analysis, and quantitative trading system
- **patterns/**: Pattern recognition and confirmation modules
- **risk/**: Risk management and scoring systems (legacy helpers; primary risk analysis now under agents/risk_analysis)
- **services/**: Various backend services for data processing and analysis
- **zerodha/**: Zerodha trading platform integration

### Key Features
- **Quantitative Trading System**: Advanced ML models and trading strategies with reorganized structure
- **Data Management**: Cached data, sector analysis, and instrument filtering
- **AI Integration**: Gemini-powered analysis and pattern detection
- **Real-time Trading**: Live trading system with risk management
- **Backtesting**: Comprehensive backtesting engine for strategy validation
- **Pattern Recognition**: Advanced pattern detection and confirmation systems

### File Counts
- **Total Python files**: ~120+ .py files
- **Configuration files**: 8 config files
- **Data files**: 310+ CSV files in cache, multiple JSON datasets
- **Model files**: 2 .joblib model files, 1 .pth PyTorch model
- **Log files**: 3 service log files
- **Virtual Environment**: Complete Python 3.10 virtual environment in myenv/
- **Prompt files**: 15 text files for AI prompts

### Recent Changes
- **Agent-Based Architecture**: Introduced `agents/` with volume, risk_analysis, mtf_analysis, and sector components
- **ML Structure Reorganization**: The `ml/quant_system/` directory has been reorganized with better separation of concerns
- **New Analysis Module**: Added dedicated `analysis/` directory for advanced analysis functionality
- **Pattern Recognition**: Moved pattern-related modules to dedicated `patterns/` directory
- **Risk Management**: Separated risk helpers under `risk/`; primary risk analysis now under `agents/risk_analysis`
- **Enhanced Data Structure**: Improved organization of data files and caching system
