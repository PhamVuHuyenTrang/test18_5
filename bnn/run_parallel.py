import threading
import datetime
import os

max_cuda1_parallel = 0
max_cuda0_parallel = 1
max_cpu_parallel = 0

num_model_list = [3]
kl_dense_list = [0.01]
kl_dropout_list = [1.]
jumpbias_list = [-2., -1.5, -1., -0.5, 0., 0.5, 1., 1.5, 2.]


cuda0_semaphore = threading.Semaphore(max_cuda0_parallel)
cuda1_semaphore = threading.Semaphore(max_cuda1_parallel)
cpu_semaphore = threading.Semaphore(max_cpu_parallel)
list_semaphore = threading.Semaphore(1)


def job(cmdd, device):
    print('running: ' + cmdd + ' --device ' + device)
    print(datetime.datetime.now())
    if device == 'cpu':
        os.system(cmdd + ' --device ' + device)
        cpu_semaphore.release()
    elif device == 'cuda':
        os.system(cmdd + ' --device ' + device)
        cuda0_semaphore.release()
    elif device == 'cuda:1':
        os.system(cmdd + ' --device ' + device)
        cuda1_semaphore.release()
    print('finish: ' + cmdd + ' --device ' + device)
    print(datetime.datetime.now())


def cpu_consumer(cmdList):
    cpu_procList = []
    while(True):
        cpu_semaphore.acquire()
        list_semaphore.acquire()
        leng = len(cmdList)
        if leng == 0:
            list_semaphore.release()
            cpu_semaphore.release()
            break
        else:
            cur_comand = cmdList[0]
            cmdList.remove(cmdList[0])
            list_semaphore.release()

            proc = threading.Thread(target=job, args=(cur_comand, 'cpu'))
            proc.start()
            cpu_procList.append(proc)

    for proc in cpu_procList:
        proc.join()


def cuda0_consumer(cmdList):
    cuda0_procList = []
    while(True):
        cuda0_semaphore.acquire()
        list_semaphore.acquire()
        leng = len(cmdList)
        if leng == 0:
            list_semaphore.release()
            cuda0_semaphore.release()
            break
        else:
            cur_comand = cmdList[0]
            cmdList.remove(cmdList[0])
            list_semaphore.release()

            proc = threading.Thread(target=job, args=(cur_comand, 'cuda'))
            proc.start()
            cuda0_procList.append(proc)

    for proc in cuda0_procList:
        proc.join()


def cuda1_consumer(cmdList):
    cuda1_procList = []
    while(True):
        cuda1_semaphore.acquire()
        list_semaphore.acquire()
        leng = len(cmdList)
        if leng == 0:
            list_semaphore.release()
            cuda1_semaphore.release()
            break
        else:
            cur_comand = cmdList[0]
            cmdList.remove(cmdList[0])
            list_semaphore.release()

            proc = threading.Thread(target=job, args=(cur_comand, 'cuda:1'))
            proc.start()
            cuda1_procList.append(proc)

    for proc in cuda1_procList:
        proc.join()


if __name__ == "__main__":
    cmdList = []

    for num_model in num_model_list:
        for kl_dropout in kl_dropout_list:
            for kl_dense in kl_dense_list:
                for jumpbias in jumpbias_list:
                    cmdList.append(
                        f'python main.py --num-models {num_model} --weight-kl-dense {kl_dense} --weight-kl-dropout {kl_dropout} --max-iter 10 --dataset svhn --variational-dropout --learning-rate  0.001 --surrogate-prior-path checkpoint\SVHNdropout\epoch16.pt --epochs 200 --diffusion 2. --changerate 3 --jump-bias {jumpbias}')

    cuda0_proc = threading.Thread(target=cuda0_consumer, args=([cmdList]))
    cuda1_proc = threading.Thread(target=cuda1_consumer, args=([cmdList]))
    cuda0_proc.start()
    cuda1_proc.start()
    cuda0_proc.join()
    cuda1_proc.join()
