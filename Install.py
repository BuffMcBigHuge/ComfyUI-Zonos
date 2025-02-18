import subprocess
import os.path


class Install:
    zonosPath = os.path.join(os.path.dirname(__file__), "Zonos")

    @staticmethod
    def check_install():
        if not os.path.exists(Install.zonosPath):
            Install.clone()
            Install.install()

    @staticmethod
    def install():
        subprocess.run(
            ['git', 'submodule', 'update', '--init', '--recursive'],
            cwd=os.path.dirname(__file__),
            shell=True,
        )

    @staticmethod
    def clone():
        subprocess.run(
            [
                "git", "clone", "--depth", "1",
                "https://github.com/Zyphra/Zonos",
                "Zonos"
            ],
            cwd=os.path.dirname(__file__)
        )
