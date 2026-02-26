# main.py
import json
import os
import webbrowser
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from utils import get_redis_instance
import websocket

redis_client = get_redis_instance()

def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        graph: The ChatbotRagRetrieval instance to be used by the app.

    Returns:
        FastAPI: The configured FastAPI application.
    """

    app = FastAPI(
        title="MultiAgent Boilerplate API",
        description="API for interacting with the MultiAgent Boilerplate",
        version="1.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve static files from assets directory
    app.mount("/assets", StaticFiles(directory="assets"), name="assets")

    @app.get("/", response_class=HTMLResponse)
    async def serve_chat() -> HTMLResponse:
        """Serve the chat interface HTML page.
        
        Returns:
            HTMLResponse: The rendered chat interface HTML
        """
        with open("assets/chat.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())

    app.include_router(websocket.router)


    @app.get("/leads_generated", response_class=HTMLResponse)
    async def serve_leads_generated() -> HTMLResponse:
        """Serve the leads generated HTML page.

        Returns:
            HTMLResponse: The rendered leads generated HTML
        """
        with open("assets/leads_generated.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())

    @app.get("/get_lead_datalist", response_class=JSONResponse)
    async def get_lead_datalist() -> JSONResponse:
        """Serve the leads generated HTML page.

        Returns:
            HTMLResponse: The rendered leads generated HTML
        """
        leads = redis_client.hgetall("leads_generated")
        # return json response
        ret = []
        for lead in leads:
            ret.append(json.loads(redis_client.hget("leads_generated", lead)))
        return JSONResponse(content=ret)



    for route in app.routes:
        methods = getattr(route, "methods", ["WEBSOCKET"])  # REST has .methods, WS doesn't
        print(f"{methods} -> {route.path}")


    return app      

def main() -> None:
    """Initialize and start the multiagent server.

    This function:
    1. Creates the Websocket instance
    2. Initializes the FastAPI application
    3. Opens the chat interface in the default web browser
    4. Starts the uvicorn server
    """
    # Initialize the Websocket instance
    # graph = ChatbotRagRetrieval()
    # Create the FastAPI application
    # fastapi_app = create_app(graph)
    webserver_port = int(os.environ.get("PORT", 8000))
    fastapi_app = create_app()
    
    # Open chat interface in browser
    # webbrowser.open(f'http://localhost:{webserver_port}')
    
    # Start the server
    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=webserver_port,
        log_level="info"
    )

if __name__ == "__main__":
    main()
