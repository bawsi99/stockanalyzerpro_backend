# Requirements Management

This document explains the requirements structure for the StockAnalyzer Pro backend.

## Files Overview

### `requirements.txt` (Production)
- **Purpose**: Production dependencies only
- **Usage**: `pip install -r requirements.txt -c constraints.txt`
- **Contents**: Core web backend, ML libraries, and utilities
- **Render-safe**: Optimized for deployment

### `requirements-dev.txt` (Development)
- **Purpose**: Development dependencies including Jupyter
- **Usage**: `pip install -r requirements-dev.txt`
- **Contents**: All production deps + Jupyter ecosystem + dev tools
- **Note**: NOT for production deployment

### `constraints.txt` (Constraints)
- **Purpose**: Explicitly exclude problematic packages
- **Usage**: Used with `-c` flag during pip install
- **Contents**: Jupyter packages set to version 0.0.0

### `runtime.txt` (Root)
- **Purpose**: Specify Python version for Render
- **Contents**: `python-3.10.14`
- **Why**: Ensures compatibility with ML library wheels

## Installation Commands

### Production (Render)
```bash
pip install -r backend/requirements.txt -c backend/constraints.txt
```

### Development (Local)
```bash
pip install -r backend/requirements-dev.txt
```

### Render Build Script
The `render_build.sh` script automatically uses the production setup.

## Key Improvements

1. **Python Version**: Fixed to 3.10.14 for wheel compatibility
2. **Clean Dependencies**: Removed all Jupyter-related packages from production
3. **CatBoost**: Uses conditional installation for Python < 3.11
4. **No Blacklisting**: Removed confusing `<0` version constraints
5. **Separate Dev Environment**: Jupyter tools only in development

## Troubleshooting

If you encounter build issues on Render:
1. Check that `runtime.txt` specifies Python 3.10.14
2. Ensure you're using the constraints file
3. Verify no Jupyter packages are in production requirements
