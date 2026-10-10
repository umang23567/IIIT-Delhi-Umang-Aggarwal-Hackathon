import os
import sys

def main():
    print("=" * 70)
    print("S&P Global & CRISIL - Enterprise Financial NLP Risk Platform")
    print("=" * 70)
    print("Launching Executive Risk Intelligence Cockpit on http://localhost:8501 ...")

    # Bypass credentials file check & disable file watchers
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

    import streamlit.web.bootstrap as bootstrap
    from streamlit import config
    import streamlit.runtime.credentials as cred

    # Mock credentials check so it never prompts in terminal
    cred._check_credential = lambda: None

    config.set_option("server.headless", True)
    config.set_option("server.port", 8501)
    config.set_option("browser.gatherUsageStats", False)
    config.set_option("server.fileWatcherType", "none")

    script_path = os.path.join(os.path.dirname(__file__), "src", "dashboard", "app.py")
    bootstrap.run(script_path, False, [], flag_options={})

if __name__ == "__main__":
    main()
