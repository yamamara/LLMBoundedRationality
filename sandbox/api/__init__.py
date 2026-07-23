def create_api_app(*args, **kwargs):
    from sandbox.api.app import create_api_app as factory

    return factory(*args, **kwargs)

__all__ = ["create_api_app"]
