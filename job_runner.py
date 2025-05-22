import subprocess
import os
import signal
import random
import platform 
import time

from apps import app as app_loc

SYSTEM = platform.node()
TEST_DIR = "./tests/"

terminate = False

MAX_JOBS = 50
MAX_QUEUE = 30
REPEAT_COUNT = 5
WAIT_TIME = 1

queued_jobs = {}
active_jobs = {}

def queue_job(index, test_path, app, command):
    """ Add a job to our Python queue.
    """
    # Ensure the queue doesn't get too big too fast.
    # Run some existing jobs until we are under the threshold.
    while len(queued_jobs) > MAX_QUEUE:
        run_job(lazy=False)

    queued_jobs[index] = {"test_path": test_path,
                          "app":      app,
                          "command":  command}
    run_job(lazy=True)
    return

def run_job(index=0, lazy=False):
    """ Run the job in SLURM/local.
    If lazy, only try queueing once.
    """
    if index != 0:
        job = queued_jobs[index]
    else:
        # If index = 0, pick a job at random from the queue.
        index, job = random.choice(list(queued_jobs.items()))

    # Wait until the test is ready to run.
    # On HPC, wait until the queue empties a bit.
    if "voltrino" in SYSTEM or "eclipse" in SYSTEM or "ghost" in SYSTEM:
        while True:
            # Get the number of jobs currently in my queue.
            n_jobs = int(subprocess.run("squeue -u kmlamar | grep -c kmlamar",
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,
                                        shell=True,
                                        check=False,
                                        encoding='utf-8').stdout)

            # If there is room on the queue, break out of the loop.
            # On my account, 5 jobs can run at once (MaxJobsPU),
            # Can check by running: sacctmgr show qos format=MaxJobsPU,MaxSubmitPU
            if n_jobs < MAX_JOBS:
                break

            if lazy:
                # If there is no room in the queue and we are lazy,
                # don't bother waiting and try again later.
                return False
            else:
                # DEBUG
                # print(str(len(queued_jobs)) + " jobs in queue.")

                # Wait before trying again.
                time.sleep(WAIT_TIME)
    # On local, do nothing.

    # Run the test case.
    # On HPC, submit the SLURM script.
    if "voltrino" in SYSTEM or "eclipse" in SYSTEM or "ghost" in SYSTEM:
        print("Queuing app: " + job["app"].name + "\t test: " + str(index))
        output = subprocess.run("sbatch submit.slurm",
                                cwd=job["test_path"],
                                shell=True,
                                check=False,
                                encoding='utf-8',
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT).stdout
        # If the output doesn't match, something went wrong.
        if "Submitted batch job " not in output:
            print(output)
            return False
        job_id = int(output.split("Submitted batch job ", 1)[1])
        # Add the queued job to our wait list.
        # We add a dictionary so we can keep track of things when we
        # handle the output later.
        active_jobs[job_id] = {"app": job["app"],
                               "index": index, "path": job["test_path"]}
    # On local, run the command.
    else:
        # print("Running app: " + job["app"] + "\t test: " + str(index))
        start = time.time()
        output = str(subprocess.run(job["command"],
                                    cwd=job["test_path"],
                                    shell=True,
                                    check=False,
                                    encoding='utf-8',
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT).stdout)
        # features[job["app"]][index]["timeTaken"] = time.time() - start
        job["app"].features[index]["timeTaken"] = time.time() - start

        # Save the command to file.
        with open(job["test_path"] / "command.txt", "w+", encoding="utf-8") as text_file:
            text_file.write(job["command"])
        # Save the output in the associated test's folder.
        with open(job["test_path"] / "output.txt", "w+", encoding="utf-8") as text_file:
            text_file.write(output)
        # features = jobscrape_output(output, job["app"], index)
        features = job["app"].scrape_output(output,index)

    # Remove the job from the list.
    queued_jobs.pop(index)

    return True

def finish_active_jobs(lazy=False):
    """ Handle any unfinished outputs.
    """
    global features

    print(len(queued_jobs), "number of jobs")
    if not lazy:
        # Ensure everything generated is in the active jobs list.
        while len(queued_jobs) > 0:
            run_job(lazy=False)

    # Keep running this loop until all active jobs have completed and been parsed.
    while len(active_jobs) > 0:
        # We want to finish a job each iteration.
        finished_a_job = False
        # Try to find a completed job in our active list.
        for job in active_jobs:
            # If the job is done, it will not be found in the queue.
            job_status = subprocess.run("squeue -j " + str(job),
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,
                                        shell=True,
                                        check=False,
                                        encoding='utf-8').stdout
            # If the job is done.
            if "Invalid job id specified" in job_status \
                    or "kmlamar" not in job_status:
                # Open the file with the completed job.
                try:
                    f = open(active_jobs[job]["path"] /
                             ("slurm-" + str(job) + ".out"), "r", encoding="utf-8")
                    output = f.read()
                except IOError:
                    # The file likely doesn't exist yet.
                    # Try again later.
                    continue
                f.close()
                # Parse the output.
                curr_job = active_jobs[job]

                features = curr_job["app"].scrape_output(
                    output, curr_job["index"])
                # Report an error to screen.
                if "error" in curr_job["app"].features[curr_job["index"]]:
                    print(str(curr_job["app"].name) + " " + str(curr_job["index"]) + ": " + str(
                        curr_job["app"].features[curr_job["index"]]["error"]))
                else:
                    print(str(curr_job["app"].name) + " " +
                          str(curr_job["index"]) + ": Completed!")
                # Save the output of this job to file.
                # append_test(active_jobs[job]["app"], active_jobs[job]["index"])
                curr_job["app"].append_test(curr_job["index"])
                # The job has been parsed. Remove it from the list.
                active_jobs.pop(job)
                # We successfully finished a job.
                finished_a_job = True
                # We have found our job in the for loop.
                # Break out and start the search again.
                break
        if finished_a_job:
            # TODO: Queue another job.
            pass
        else:  # If we went through the whole queue and all remaining jobs were still active.
            # If we are lazily finishing jobs.
            if lazy:
                # Don't bother waiting. Break out now and come back later.
                break
            # Print the contents of the remaining queue.
            print(subprocess.run("squeue -u kmlamar",
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 shell=True,
                                 check=False,
                                 encoding='utf-8').stdout)
            # Wait before trying again.
            time.sleep(WAIT_TIME)

def exit_gracefully(signum, frame):
    """ Override the default signal handler to allow for graceful termination.
    ProxyAppPredictor will finish what's in its queue before closing.
    """
    global terminate 
    original_sigint = signal.getsignal(signal.SIGINT)
    # Restore the original signal handler as otherwise bad things will happen
    # in input when CTRL+C is pressed, and our signal handler is not re-entrant.
    signal.signal(signal.SIGINT, original_sigint)
    try:
        if input("\nReally quit? (y/n)> ").lower().startswith("y"):
            terminate = True
    except KeyboardInterrupt:
        print("Quitting...")
        terminate = True
    # Restore the exit gracefully handler here.
    signal.signal(signal.SIGINT, exit_gracefully)

def get_next_index(app_name):
    """ Get the next unused test index of the associated app.
    Enables testing to resume while giving each test a unique index.
    """
    try:
        idx = int(max(os.listdir(TEST_DIR + app_name + "/"))) + 1
    except FileNotFoundError:
        idx = 0
    return idx

def main():
    """ Run random permutations of tests.
    This extra variety helps training.
    """
    global terminate
    signal.signal(signal.SIGINT, exit_gracefully)
    
    # 449
    enabled_apps = [
        app_loc.Nekbone("timeTaken","./tests/nekbonedataset.csv"),
        # app.HACC_IO("timeTaken","./tests/HACC-IOdataset.csv"),
        # app.SWFFT("timeTaken","./tests/SWFFTdataset.csv"),
        # app.ExaMiniMD("timeTaken","./tests/ExaMiniMDsnapdataset.csv"),
    ]
    print("found jobs about to run")

    # Cancel via Ctrl+C.
    # While we have not canceled the test:
    while not terminate:
        # Pick a random app.
        app = random.choice(list(enabled_apps))
        # Get the parameters.
        params = app.get_params()
        # Run each test multiple times.
        for i in range(REPEAT_COUNT):
            # Get the index to save the test files.
            index = get_next_index(app.name)
            # Run the test.
            command,test_path = app.generate_test(params, index)

            queue_job(index,test_path,app,command)
        # Try to finish jobs.
        finish_active_jobs(lazy=True)
    # If we want to terminate, we can't be lazy. Be sure all jobs complete.
    finish_active_jobs(lazy=False)
    # signal.signal(signal.SIGINT, original_sigint)

main()