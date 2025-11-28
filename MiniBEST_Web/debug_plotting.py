import sys
import os
sys.path.append(os.getcwd())
try:
    import web_backend.analysis.plotting as p
    print("Module imported successfully")
    if 'generate_plot' in dir(p):
        print("generate_plot found")
    else:
        print("generate_plot NOT found")
        print("Symbols:", dir(p))
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()

