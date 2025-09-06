# Backend Directory Tree

```
backend/
├── __init__.py
├── README.md
├── api/
│   ├── __init__.py
│   ├── endpoints.py
│   ├── main.py
│   └── responses.py
├── config/
│   ├── __init__.py
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
│   │   │   ├── __init__.cpython-310.pyc
│   │   │   ├── config.cpython-310.pyc
│   │   │   ├── regimes.cpython-310.pyc
│   │   │   ├── schema.cpython-310.pyc
│   │   │   └── scoring.cpython-310.pyc
│   │   ├── config.py
│   │   ├── regimes.py
│   │   ├── schema.py
│   │   ├── scoring.py
│   │   └── weights_config.json
│   └── zerodha_instruments.csv
├── gemini/
│   ├── __pycache__/
│   │   ├── context_engineer.cpython-310.pyc
│   │   ├── debug_config.cpython-310.pyc
│   │   ├── debug_logger.cpython-310.pyc
│   │   ├── error_utils.cpython-310.pyc
│   │   ├── gemini_client.cpython-310.pyc
│   │   ├── gemini_core.cpython-310.pyc
│   │   ├── image_utils.cpython-310.pyc
│   │   ├── parallel_pattern_detection.cpython-310.pyc
│   │   ├── prompt_manager.cpython-310.pyc
│   │   ├── schema.cpython-310.pyc
│   │   └── token_tracker.cpython-310.pyc
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
├── ml/
│   ├── __init__.py
│   ├── __pycache__/
│   │   ├── __init__.cpython-310.pyc
│   │   ├── inference.cpython-310.pyc
│   │   └── technical_indicators.cpython-310.pyc
│   ├── advanced_analysis.py
│   ├── agent_capabilities.py
│   ├── analysis_datasets.py
│   ├── backtesting.py
│   ├── bayesian_scorer.py
│   ├── enhanced_mtf_analysis.py
│   ├── enhanced_sector_classifier.py
│   ├── heavy_load_analysis.py
│   ├── inference.py
│   ├── market_regime.py
│   ├── mtf_analysis_utils.py
│   ├── pattern_confirmation.py
│   ├── pattern_database.py
│   ├── patterns/
│   │   ├── __pycache__/
│   │   ├── recognition.py
│   │   └── visualization.py
│   ├── quant_system/
│   │   ├── advanced_models/
│   │   │   ├── advanced_feature_engineer.py
│   │   │   ├── advanced_trading_system.py
│   │   │   ├── advanced_training_strategies.py
│   │   │   ├── dynamic_ensemble_manager.py
│   │   │   ├── meta_learning_framework.py
│   │   │   ├── multimodal_fusion_model.py
│   │   │   ├── nbeats_model.py
│   │   │   ├── neural_architecture_search.py
│   │   │   ├── phase2_integration_manager.py
│   │   │   ├── real_time_data_integrator.py
│   │   │   └── temporal_fusion_transformer.py
│   │   ├── catboost_info/
│   │   │   ├── catboost_training.json
│   │   │   ├── learn/
│   │   │   │   └── events.out.tfevents
│   │   │   ├── learn_error.tsv
│   │   │   ├── time_left.tsv
│   │   │   └── tmp/
│   │   ├── datasets/
│   │   │   ├── robust_patterns_meta.json
│   │   │   ├── robust_patterns.parquet
│   │   │   ├── test_general_patterns_meta.json
│   │   │   └── test_general_patterns.parquet
│   │   ├── ml/
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__/
│   │   │   ├── core.py
│   │   │   ├── feature_engineering.py
│   │   │   ├── hybrid_ml.py
│   │   │   ├── pattern_ml.py
│   │   │   ├── raw_data_ml.py
│   │   │   ├── traditional_ml.py
│   │   │   └── unified_manager.py
│   │   ├── models/
│   │   │   ├── enhanced_pattern_model.joblib
│   │   │   ├── pattern_catboost.joblib
│   │   │   └── pattern_registry.json
│   │   ├── nas_results/
│   │   │   ├── nas_results_1755854300.json
│   │   │   └── nas_results_1755854599.json
│   │   ├── advanced_trading_strategies_with_risk_management.py
│   │   ├── analyze_price_ml.py
│   │   ├── backtesting_engine.py
│   │   ├── best_nbeats_model.pth
│   │   ├── data_pipeline.py
│   │   ├── dataset_builder.py
│   │   ├── deploy_live_trading_system.py
│   │   ├── enhanced_data_pipeline.py
│   │   ├── enhanced_feature_engineering.py
│   │   ├── enhanced_ml_engine.py
│   │   ├── enhanced_training.py
│   │   ├── live_trading.log
│   │   ├── model_comparison.py
│   │   ├── production_market_data_integration.py
│   │   ├── production_trading_system.py
│   │   ├── production_trading.log
│   │   ├── quant_system_integration.py
│   │   ├── risk_management.py
│   │   ├── robust_evaluation.py
│   │   ├── run_quant_tests.py
│   │   ├── simplified_production_trading_system.py
│   │   ├── train_evaluate_pattern.py
│   │   └── zerodha_instruments.csv
│   ├── risk_scoring.py
│   ├── sector_benchmarking.py
│   ├── sector_classifier.py
│   ├── service_specific_memory_analyzer.py
│   ├── technical_indicators.py
│   └── volume_profile.py
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
│   ├── analysis_service.py
│   ├── central_data_provider.py
│   ├── consolidated_service.py
│   ├── data_service.py
│   ├── database_service.py
│   ├── enhanced_data_service.py
│   └── websocket_service.py
└── zerodha/
    ├── __init__.py
    ├── client.py
    ├── token.py
    └── ws_client.py
```

## Summary

The backend directory contains a comprehensive stock analysis and trading system with the following main components:

### Core Modules
- **api/**: FastAPI endpoints and responses
- **config/**: Configuration files and requirements
- **core/**: Core utilities, cache management, and client connections
- **data/**: Data storage, analysis datasets, and sector categorization
- **gemini/**: AI/ML integration with Google Gemini
- **ml/**: Machine learning models, analysis, and quantitative trading system
- **services/**: Various backend services for data processing and analysis
- **zerodha/**: Zerodha trading platform integration

### Key Features
- **Quantitative Trading System**: Advanced ML models and trading strategies
- **Data Management**: Cached data, sector analysis, and instrument filtering
- **AI Integration**: Gemini-powered analysis and pattern detection
- **Real-time Trading**: Live trading system with risk management
- **Backtesting**: Comprehensive backtesting engine for strategy validation

### File Counts
- **Total Python files**: ~100+ .py files
- **Configuration files**: 8 config files
- **Data files**: 280+ CSV files in cache, multiple JSON datasets
- **Model files**: 2 .joblib model files, 1 .pth PyTorch model
- **Log files**: 2 trading log files
- **Virtual Environment**: Complete Python 3.10 virtual environment in myenv/
