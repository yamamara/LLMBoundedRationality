from a2wsgi import WSGIMiddleware

from frontend import create_dash_app
from sandbox.api import create_api_app


app = create_api_app()
dash_app = create_dash_app()
app.mount("/", WSGIMiddleware(dash_app.server))
