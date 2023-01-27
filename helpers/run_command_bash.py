import subprocess


class RunCommandBash:
    def __init__(self, cmd, args):
        self.cmd = cmd
        self.args = args

    def __call__(self):
        print("Running command: " + self.cmd + " " + " ".join(self.args))
        subprocess.run([self.cmd, *self.args])