from prefect import flow, task
import subprocess

@task(retries=2, retry_delay_seconds=30)
def run(cmd:list[str]):
    print(">", " ".join(cmd)); subprocess.run(cmd, check=True)

@flow(name="HAR MLOps")
def har_flow():
    run(["dvc","repro","-f"])

if __name__=="__main__":
    har_flow()
