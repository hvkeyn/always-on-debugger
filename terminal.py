import subprocess
import sys
import os
from debugger import Debugger

class Command:
    def __init__(self, args: list[str]):
        if len(args) < 1:
            print("Usage: python terminal.py <script_filename> [arguments...]")
            exit(1)
        self.raw_command = " ".join(args)
        self.filename = args[0] if len(args) > 0 else None
        self.args = args[1:]  # The remaining arguments are passed to the script
        self.debugger = Debugger()

        if not self.filename:
            raise ValueError("Filename not provided! Please specify a script to execute.")

    def run(self):
        try:
            print("Subprocess start...")

            # Check if the file is provided and exists
            script_path = os.path.abspath(self.filename)
            if not os.path.exists(script_path):
                raise FileNotFoundError(f"File not found: {script_path}")

            # Execute the file
            command = ["python", script_path] + self.args
            result = subprocess.run(
                command,
                check=False,  # Allow the process to return non-zero exit codes
                text=True,
                capture_output=True
            )

            # Print stdout and stderr
            print(f"stdout: {result.stdout.strip()}")
            print(f"stderr: {result.stderr.strip()}")

            if result.returncode != 0 or result.stderr.strip():
                print("Subprocess failed with an error, initiating debugging...")
                self.debug_process(result.stderr)
            elif result.stdout.strip():
                print("Subprocess produced output but no errors detected.")
            else:
                print("Subprocess completed without output or errors.")
        except Exception as ex:
            print(f"An unexpected error occurred during subprocess execution: {ex}")

    def debug_process(self, error_message: str):
        print("Starting debug process...")
        filepath = self.debugger.detect_file_path(self.raw_command, error_message)
        code_snippet = None
        if filepath and os.path.exists(filepath):
            with open(filepath, 'r') as file:
                code_snippet = file.read()
        response = self.debugger.debug(self.raw_command, error_message, code_snippet)
        print(response.get("recommendation", "No recommendation available."))

if __name__ == "__main__":
    cmd = Command(sys.argv[1:])
    cmd.run()