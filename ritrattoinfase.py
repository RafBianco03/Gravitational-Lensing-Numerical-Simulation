import numpy as np
import scipy.interpolate as interp
import scipy.optimize as op
import scipy.integrate as integ
import matplotlib.pyplot as plt
import numpy.linalg as lin
import scipy.interpolate as interp
import statistics as stat

L = 5
E = 3
A = 1

m = lambda r: A*np.array([np.log(1+r)-r/(1+r)]) #Oppenheimer-Volkoff
pot = lambda r: -A*np.log(1+r)/r #Potenziale Gravitazionale NFW

fig, ax = plt.subplots(1,1)
ax.set_ylabel("$\dot{r}$ ($r_s$ unit)")
ax.set_xlabel("r ($r_s$ unit)")

r = np.linspace(0,10,10000)

dotm = lambda r: A*r/(1+r)**2 #Derivata di m(r) rispetto ad r
dotr = lambda r: np.sqrt((1-2*m(r)/r)*(E**2*np.e**(-2*pot(r)) - L**2/r**2)) #Derivata di r rispetto a λ
ddotr = lambda r: (1-2*m(r)/r)*(-np.e**(2*pot(r))*m(r)/r**2*E**2 - (dotm(r)/r - m(r)/r**2)*dotr(r)**2 + L**2/r**3) #Derivata seconda di r rispetto a λ

Ln = np.linspace(0,5,100)
for i in range(len(Ln)):
    L = Ln[i]
    yn = dotr(r)[0]
    #Plot delle orbite nello spazio delle fasi
    pl = ax.plot(r,yn)
    ax.plot(r,-yn, color = pl[0].get_color())

    if i == len(Ln)-1:
        rn = np.linspace(0,10,100)
        yn = dotr(rn)[0]
        #Plot delle velocità vettoriali di un'orbita.
        ax.quiver(rn,yn,dotr(rn),ddotr(rn), units = "xy")
        ax.quiver(rn,-yn,-dotr(rn),ddotr(rn), units = "xy")
plt.grid()
plt.show()
