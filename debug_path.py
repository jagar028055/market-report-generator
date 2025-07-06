import os
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

print("--- Starting Debug Script ---")

# 1. Print Current Working Directory
cwd = os.getcwd()
print(f"[1] Current Working Directory: {cwd}")

# 2. Construct path to templates directory
# Assuming this script is in the repository root, like main.py
templates_dir = os.path.join(cwd, 'templates')
print(f"[2] Constructed 'templates' directory path: {templates_dir}")

# 3. Check if the templates directory exists and is a directory
is_dir = os.path.isdir(templates_dir)
print(f"[3] Does the 'templates' directory exist? {is_dir}")

# 4. If it exists, list its contents
if is_dir:
    try:
        files = os.listdir(templates_dir)
        print(f"[4] Files in 'templates' directory: {files}")
    except Exception as e:
        print(f"[4] Error listing files in 'templates' directory: {e}")
else:
    print("[4] Skipping file listing because directory does not exist.")

# 5. Attempt to initialize Jinja2 and load the template
print("[5] Attempting to load 'report_template.html' with Jinja2...")
if is_dir:
    try:
        env = Environment(loader=FileSystemLoader(templates_dir))
        template = env.get_template('report_template.html')
        print("[5] SUCCESS: 'report_template.html' loaded successfully.")
    except TemplateNotFound as e:
        print(f"[5] FAILED: jinja2.exceptions.TemplateNotFound: {e}")
    except Exception as e:
        print(f"[5] FAILED: An unexpected error occurred: {e}")
else:
    print("[5] Skipping Jinja2 load attempt because directory does not exist.")

print("--- Debug Script Finished ---")
