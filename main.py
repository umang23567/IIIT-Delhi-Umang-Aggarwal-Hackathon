"""CLI entry point.  python main.py all   |  gen-data  |  brief  |  analyze "text" """
import argparse
import json


def main():
    ap = argparse.ArgumentParser(description="ContagionLens")
    ap.add_argument("cmd", choices=["all", "gen-data", "brief", "analyze"])
    ap.add_argument("text", nargs="?", default="")
    ap.add_argument("--regen", action="store_true", help="regenerate synthetic data")
    a = ap.parse_args()
    if a.cmd == "gen-data":
        from src.data_gen import generate
        print("news, tweets:", generate())
    elif a.cmd == "all":
        from src.pipeline import run_all
        v = run_all(regen=a.regen)
        print(json.dumps(v["backtest"], indent=1))
        print("IC:", json.dumps(v["ic"]))
        print("Outputs written to ./output  |  dashboard: streamlit run src/dashboard.py")
    elif a.cmd == "brief":
        print(open("output/brief.md").read())
    elif a.cmd == "analyze":
        from src.nlp import analyze
        r = analyze(a.text, "news")
        print(json.dumps(r, indent=1, default=str))


if __name__ == "__main__":
    main()
