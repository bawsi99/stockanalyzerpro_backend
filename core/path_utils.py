#!/usr/bin/env python3
"""
Path Utilities for Production-Safe Path Resolution

This module provides centralized path resolution utilities that work
in both local development and production environments like Render.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class PathResolver:
    """
    Centralized path resolver that handles different deployment environments.
    """
    
    _instance = None
    _backend_root = None
    
    def __new__(cls):
        """Singleton pattern to ensure consistent path resolution."""
        if cls._instance is None:
            cls._instance = super(PathResolver, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._backend_root = self._find_backend_root()
            logger.info(f"PathResolver initialized with backend root: {self._backend_root}")
    
    def _find_backend_root(self) -> str:
        """
        Find the backend root directory using multiple strategies.
        Returns the backend directory path.
        """
        # Get the current file's directory (core/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try multiple path strategies for different deployment scenarios
        possible_backends = [
            # Local development: core/../ -> backend/
            os.path.join(current_dir, '..'),
            # Render deployment paths
            '/opt/render/project/src/backend',
            '/opt/render/project/backend',
            # Docker/Heroku style paths
            '/app/backend',
            '/app/src/backend',
            # CWD-based paths
            os.path.join(os.getcwd(), 'backend'),
            os.path.join(os.getcwd(), 'src', 'backend'),
            # Environment variable override
            os.environ.get('BACKEND_ROOT', ''),
            os.environ.get('PROJECT_ROOT', ''),
            # Split on known directory names
            current_dir.split('backend')[0] + 'backend' if 'backend' in current_dir else '',
            current_dir.split('core')[0] if 'core' in current_dir else '',
        ]
        
        # Filter out empty strings
        possible_backends = [path for path in possible_backends if path]
        
        backend_dir = None
        for path in possible_backends:
            if os.path.exists(path) and os.path.isdir(path):
                # Verify it's actually a backend directory by checking for common files/directories
                backend_indicators = [
                    os.path.join(path, 'config'),
                    os.path.join(path, 'data'),
                    os.path.join(path, 'zerodha'),
                    os.path.join(path, 'services'),
                    os.path.join(path, 'core'),
                ]
                
                if any(os.path.exists(indicator) for indicator in backend_indicators):
                    backend_dir = os.path.abspath(path)
                    logger.info(f"Found backend directory: {backend_dir}")
                    break
        
        if not backend_dir:
            # Fallback: use current directory's parent
            backend_dir = os.path.abspath(os.path.join(current_dir, '..'))
            logger.warning(f"Using fallback backend directory: {backend_dir}")
        
        return backend_dir
    
    def get_backend_root(self) -> str:
        """Get the backend root directory."""
        return self._backend_root
    
    def get_config_path(self, filename: str = '.env') -> str:
        """Get path to config file."""
        return os.path.join(self._backend_root, 'config', filename)
    
    def get_data_path(self, filename: str = '') -> str:
        """Get path to data file or directory."""
        if filename:
            return os.path.join(self._backend_root, 'data', filename)
        return os.path.join(self._backend_root, 'data')
    
    def get_zerodha_instruments_csv_path(self) -> str:
        """Get path to Zerodha instruments CSV file."""
        return self.get_data_path('zerodha_instruments.csv')
    
    def ensure_directory_exists(self, path: str) -> bool:
        """
        Ensure a directory exists, creating it if necessary.
        Returns True if directory exists or was created successfully.
        """
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except (OSError, PermissionError) as e:
            logger.warning(f"Could not create directory {path}: {e}")
            return False
    
    def is_file_writable(self, file_path: str) -> bool:
        """Check if a file path is writable."""
        try:
            # Check if we can write to the directory
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                return self.ensure_directory_exists(directory)
            
            # Check if file exists and is writable
            if os.path.exists(file_path):
                return os.access(file_path, os.W_OK)
            
            # Check if we can create a new file in the directory
            return os.access(directory, os.W_OK)
        except Exception:
            return False
    
    def get_safe_write_path(self, preferred_path: str, fallback_dir: Optional[str] = None) -> str:
        """
        Get a safe path for writing files, with fallback options.
        
        Args:
            preferred_path: The preferred file path
            fallback_dir: Optional fallback directory (defaults to temp)
        
        Returns:
            A writable file path
        """
        # Try the preferred path first
        if self.is_file_writable(preferred_path):
            return preferred_path
        
        # Try fallback directory
        if fallback_dir:
            filename = os.path.basename(preferred_path)
            fallback_path = os.path.join(fallback_dir, filename)
            if self.is_file_writable(fallback_path):
                logger.info(f"Using fallback path: {fallback_path}")
                return fallback_path
        
        # Last resort: use temp directory
        import tempfile
        filename = os.path.basename(preferred_path)
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        logger.warning(f"Using temporary path: {temp_path}")
        return temp_path


# Global instance
path_resolver = PathResolver()


def get_backend_root() -> str:
    """Get the backend root directory."""
    return path_resolver.get_backend_root()


def get_config_path(filename: str = '.env') -> str:
    """Get path to config file."""
    return path_resolver.get_config_path(filename)


def get_data_path(filename: str = '') -> str:
    """Get path to data file or directory."""
    return path_resolver.get_data_path(filename)


def get_zerodha_instruments_csv_path() -> str:
    """Get path to Zerodha instruments CSV file."""
    return path_resolver.get_zerodha_instruments_csv_path()


def get_safe_write_path(preferred_path: str, fallback_dir: Optional[str] = None) -> str:
    """Get a safe path for writing files, with fallback options."""
    return path_resolver.get_safe_write_path(preferred_path, fallback_dir)


def ensure_directory_exists(path: str) -> bool:
    """Ensure a directory exists, creating it if necessary."""
    return path_resolver.ensure_directory_exists(path)