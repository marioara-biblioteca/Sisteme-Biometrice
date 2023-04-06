import scipy.io
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms as T


from pyeer.eer_info import get_eer_stats
from pyeer.report import generate_eer_report, export_error_rates
from pyeer.plot import plot_eer_stats

#comparam detali extrase din imagini, nu imaginile cu totul
#de ex crestele amprentelor
legitimi = scipy.io.loadmat('gms.mat')['gms']
impostori = scipy.io.loadmat('ims.mat')['ims']

#Calculați media și deviația standard pentru scorurile utilizatorilor legitimi și, respectiv scorurile impostorilor.
def cerinta1():
    print("\nMean value for legitimi: " +str(np.mean(legitimi)) + "\nStd value for legitimi: "+str(np.std(legitimi)))
    print("\nMean value for legitimi: " +str(np.mean(impostori)) + "\nStd value for legitimi: "+str(np.std(impostori)))

#Reprezentați grafic distribuțiile scorurilor utilizatorilor legitimi și ale impostorilor.
def cerinta2():
    # plt.hist(legitimi,100)
    # plt.hist(impostori,100)
    values, counts = np.unique(legitimi, return_counts=True)
    plt.plot(values,counts)
    values, counts = np.unique(impostori, return_counts=True)
    plt.plot(values,counts)
    plt.title('Score')
    plt.show()

#Reprezentați grafic ratele erorilor de potrivire falsă FMR și de nepotrivire falsă FNMR în funcție de pragul de decizie t.
sampling=np.linspace(0,1)
fmr=[]
fnmr=[]
for t in sampling:
    count = sum(map(lambda x : x>=t , impostori))[0]
    fmr.append(count/len(impostori))
    count=sum(map(lambda x : x<t , legitimi))[0]
    fnmr.append(count/len(legitimi))
def cerinta3():
    fig = plt.figure(figsize=(10,10))
    plt.plot(sampling,fnmr,label='FNMR')
    plt.plot(sampling,fmr,label='FMR')
    plt.title("FMR/FNMR")
    plt.legend(loc="upper right")
    plt.show()
    plt.close()
def cerinta4():
    fig = plt.figure(figsize=(10,10))
    plt.plot(fmr,fnmr)
    plt.title("DET curve")
    plt.xlabel("FMR(t)")
    plt.ylabel("FNMR(t)")
    plt.show()
#scorurile p acc persona sunt mai mari ca scorurile pt personae diferite
#cele pt pers diferite sunt mai aproape std si mean
#o sa trb sa setez un prag ca sa pot zice daca persoana e impostor sau nu
#cerinta1()
cerinta3()
#cerinta3()
# cerinta4()