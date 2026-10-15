"""Release Chroma's native file handles before removing temporary databases on Windows."""

from contextlib import contextmanager
import tempfile
import gc
import time

from chromadb.api.shared_system_client import SharedSystemClient


def stop_test_databases():
    # Chroma caches native clients globally; a Python reference disappearing does
    # not close their mmap files. This touches only clients in the test process.
    for system in list(SharedSystemClient._identifier_to_system.values()):
        system.stop()
    SharedSystemClient.clear_system_cache()
    gc.collect()


def cleanup_test_directory(directory):
    # Native SQLite shutdown may finish just after the Rust object is dropped.
    for attempt in range(50):
        try:
            directory.cleanup()
            return
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(0.1)


@contextmanager
def temporary_database_directory():
    with tempfile.TemporaryDirectory() as directory:
        try:
            yield directory
        finally:
            import streamlit as st

            st.cache_resource.clear()
            st.cache_data.clear()
            stop_test_databases()
