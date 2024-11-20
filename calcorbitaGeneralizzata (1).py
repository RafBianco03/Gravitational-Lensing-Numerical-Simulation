import numpy as np
import scipy.interpolate as interp
import scipy.optimize as op
import scipy.integrate as integ
import matplotlib.pyplot as plt
import numpy.linalg as lin
import scipy.interpolate as interp
import statistics as stat

#Costanti del moto
xi = 0
L = 0
E = 3
A = 1

#Numero di punti generati per la rappresentazione di ogni traiettoria luce
prec = int(1e4)
err = 1/prec

#Qui sono presenti tutte le funzioni di cui ho bisogno per la simulazione
m = lambda r: A*np.array([np.log(1+r)-r/(1+r)]) #Oppenheimer-Volkoff
dr = lambda r: -np.array(np.maximum([ ((1+r)**(A/r))*E**2 -(L)**2/r**2 ],0))**(1/2)*np.array(np.maximum([(1-2*np.array(m(r))/r)],0))**(1/2) #Oppenheimer-Volkoff
def dr_(r):
    if (((1+r)**(A/r))*E**2 -(L)**2/r**2 < 0).any():
        print("NO")
        return 0
    if (1-2*np.array(m(r))/r < 0).any():
        print("NO^2")
        return 0
    return -np.array(np.maximum(((1+r)**(A/r))*E**2 -(L)**2/r**2,0))**(1/2)*np.array(np.maximum((1-2*np.array(m(r))/r),0))**(1/2) #Oppenheimer-Volkoff
pot = lambda r: -A*np.log(1+r)/r
dphi = lambda r: L*r**(-2)
dt = lambda r: -E*np.e**(-2*pot(r))
dr_dt = lambda t,r: -(dr(r)/dt(r))[0][0]
dr_dphi = lambda t,r: np.sign(L)*dr(r)/dphi(r)[0]
ddr = lambda r: (m(r)/(r*(r-2*m(r))))*(dt(r)**2) - ((1-2*m(r)/r)**(-1))*(A*((1+r)**(-2))-m(r)/r**2)*(dr(r)[0])**2 + (1-2*m(r)/r)*((L**2)/(r**3))
dphi_dr = lambda r: L/((dr(r))*r**2)

#Raggio minimo(parametro di impatto) e raggio minimo "buco nero"
eq = lambda r: np.e**(-2*pot(r))*r**2 - (L/E)**2
eq1 = lambda r: r - 2*m(r)[0]
OC = 0

def r(a,b,d): #Voglio calcolare r(φ), avendo dr/dφ
    ang1 = np.linspace(a,b,prec) #Prendo un linspace di angoli arbitrario
    #Metodo solve_ivp
    ang_span = [a,b] #Intervallo di angoli
    sol = integ.solve_ivp(dr_dphi, ang_span, [d], dense_output=True) #Genera la soluzione all'equazione differenziale, sugli angoli ang_span e un raggio iniziale r(a)=d
    r = sol.sol(ang1)[0] #Genero la soluzione sullo span

    #if (((1+r)**(A/r))*E**2 -(L)**2/r**2 < 0).any():
    #    print("problema")
    if ((1-2*np.array(m(r))/r) < 0).any():
        return None

    #togliamo le parti con velocità nulla(dato che la velocità radiale non raggiungerà mai esattamente 0)
    r_stop = np.abs(dr_dphi(0,r)[0][0]) == 0
    r = np.delete(r,r_stop) #Tolgo la parte della traiettoria a velocità nulla
    index_stop = len(r)
    b = ang1[index_stop-1]
    inter = np.abs(a-b)
    ang2 = np.linspace(b,b+inter,index_stop) #Genero il secondo intervallo di angoli, adiacente al primo(considerando il CUT)
    ang1 = np.delete(ang1, range(index_stop, len(ang1)))

    w = np.zeros(len(r)) #andamento simmetrico dopo il cambio di segno, applicato agli angoli ang2 appena creati
    for i in range(len(r)):
        w[i] = r[len(r)-1-i]

    return np.concat((r,w)), np.sign(L)*np.concat((ang1,ang2)) #Unire le due soluzioni simmetriche in una

def star(A,E): #Luce proveniente da una sola sorgente che emette isotropicamente
    global L
    global OC
    
    n = 20000 #Numero di fotoni
    rs = [] #Raggio vettore delle soluzioni
    ang = [] #Angoli delle soluzioni

    angolo = np.pi #Scelgo l'angolo iniziale
    dist_max = 10

    alfa = [] #Angoli di intercetta tra la geodetica luce e la linea di vista (L.O.S.)
    racc = [[],[]] #Informazioni di raggio di intercetta e momento angolare L

    L_max = E*dist_max*np.e**(-pot(dist_max))
    Ls = np.linspace(0.001,L_max,n) #Creo uno span di momenti angolari tra due estremi

    for i in range(len(Ls)): #Plot di due parti simmetriche rispetto all'asse x corrispondenti a momenti angolari positivi e negativi
        L = Ls[i] #Assegno il momento angolare per il ciclo
        try:
            a,b = r(-angolo, np.pi-angolo,dist_max) #Span di angoli

            #Multiimmagini a raggio:
            interferenza = np.abs(-b+xi) < err #Quando l'argomento angolare è nullo, ci troviamo sulla linea di vista L.O.S.

            #Campo vettoriale quadrivettore K
            prima = -dr(a)[0][0]
            seconda = L/a**2
            #Angolo d'inclinazione del vettore d'onda al momento dell'intercetta
            #phit = np.arctan((np.sin(b[interferenza])*prima[interferenza]+a[interferenza]*np.cos(b[interferenza])*seconda[interferenza])/(prima[interferenza]*np.cos(b[interferenza])-a[interferenza]*np.sin(b[interferenza])*seconda[interferenza]))
            phit = np.arctan((a[interferenza]*seconda[interferenza])/(prima[interferenza]))

            #Raccolgo le informazioni dell'onda i-esima
            if(len(a[interferenza])!=0):
                alfa.append(phit[0]*180/np.pi)
                racc[0].append(a[interferenza][0])
                racc[1].append(L)
                OC += 1

            L *= -1 #Plot identico speculare all'asse x
            c,d = r(-angolo, np.pi-angolo,dist_max)
            interferenza = np.abs(d-xi) < err #Quando l'argomento angolare è nullo, ci troviamo sulla linea di vista L.O.S.
            prima = -dr(c)[0][0]
            seconda = L/c**2
            phit = np.arctan((c[interferenza]*seconda[interferenza])/(prima[interferenza]))
            #Raccolgo le informazioni dell'onda i-esima
            if(len(c[interferenza])!=0):
                alfa.append(phit[0]*180/np.pi)
                racc[0].append(c[interferenza][0])
                racc[1].append(L)
                OC += 1

            rs.append(a)
            rs.append(c)
            ang.append(b)
            ang.append(d)
        except:
            continue
        
    index = [] #Controlla se c'è interferenza tra due onde sulla linea di vista
    #err = Lunghezza di scala dell'osservatore sul L.O.S. (se due raggi intercettano entro questa distanza sono considerati interferenti)
    for i in range(len(racc[0])):
        c = False
        for j in range(len(racc[0])):
            if np.abs(racc[0][i] - racc[0][j]) < err and racc[1][i] != racc[1][j]:
                #Abbiamo immagini multiple
                c = True
                #print("Interferenza tra le onde: L1:" + str(racc[1][i]) + " e L2:" + str(racc[1][j]) + " al raggio " + str(racc[0][i]))
        if not c:
            index.append(i)
    #if(len(index) == len(racc[0])):
    #    print(racc[0])
    racc[0] = list(np.delete(racc[0], index))
    racc[1] = list(np.delete(racc[1], index))
    alfa = list(np.delete(alfa, index))
    return rs,ang,racc,alfa

"""
#Rappresentazione grafica iniziale
stella = star(A,E)
rs,ang,racc,alfa = stella[0],stella[1],stella[2],stella[3]

#Ora abbiamo (t,r,θ,φ), rappresentiamo l'orbita in 2D
for i in range(len(rs)):
    x = rs[i]*np.cos(ang[i])
    y = rs[i]*np.sin(ang[i])
    plt.plot(x,y)
plt.grid()
plt.axis("equal")
raggio_x = []
raggio_y = []
xs = np.linspace(0,10,100)
for i in range(len(xs)):
    raggio_x.append(xs[i]*np.cos(xi))
    raggio_y.append(xs[i]*np.sin(xi))
plt.plot(np.array(raggio_x),np.array(raggio_y),"red")
plt.show()
"""

xi = -0.35
#Studio alpha_E(xi,r)
angs = np.linspace(xi,0,6)
c = 1
for i in range(len(angs)):
    dr = lambda r: -np.array(np.maximum([ ((1+r)**(A/r))*E**2 -(L)**2/r**2 ],0))**(1/2)*np.array(np.maximum([(1-2*np.array(m(r))/r)],0))**(1/2) #Oppenheimer-Volkoff

    xi = angs[i]
    stella = star(A,E)
    rs,ang,racc,alfa = stella[0],stella[1],stella[2],stella[3]

    fig, ax = plt.subplots(2,1)
    plotAnello = ax[0]
    plotL = ax[1]

    #Ordino i dati
    rn = racc[0]
    alpha_ = []
    rn_ = []
    ind = []
    fluct = []
    for i in range(len(rn)):
        index = []
        for j in range(len(rn)):
            if np.abs(rn[i] - rn[j]) < err and i != j and j not in ind:
                index.append(j) #abbiamo interferenza
        if len(index) > 0:
            #print(str(alfa[i]) + " " + str(alfa[index[0]]))
            alpha_.append(np.abs(alfa[i]-alfa[index[0]])) #Distanza angolare tra i due raggi interferenti -> diametro angolare anello
            ind.append(index[0])
            ind.append(i)
            rn_.append(rn[i])
            if racc[1][i]*racc[1][j] >= 0: #Prendo le interferenze con lo stesso segno nel momento angolare L
                #Stesso lato di provenienza del segnale
                fluct.append(len(rn_)-1)
    #Plot del diametro dell'anello di Einstein al raggio L.O.S.E.(line of sight event)
    plotAnello.set_ylabel("Diametro angolare $α_D$(r) (°)")
    plotAnello.set_xlabel("Distanza dalla lente r ($r_{s}$ unit)")
    plotAnello.plot(rn_,alpha_,"o", label = "$ξ$ = " + str(int(xi*(180/np.pi))))
    plotAnello.legend(loc = "upper right")
    plotL.set_xlabel("Distanza dalla lente r ($r_{s}$ unit)")
    plotL.set_ylabel("L(r)")
    plotL.plot(racc[0],racc[1],"o")
    plt.savefig("C:/Users/raf/Documents/Programmi/tesi/generalizzata/plot" + str(int(xi*(180/np.pi))) + ".png")

    print(c)#Conta giri
    c+=1

    fig, pfluct = plt.subplots(1,1)
    pfluct.set_xlabel("Distanza dalla lente r ($r_{s}$ unit)")
    pfluct.set_ylabel("$α_D$(r)")
    
    for i in range(len(fluct)):
        pfluct.plot(rn_[fluct[i]],alpha_[fluct[i]],"o")

    
    V = []
    for i in range(len(rn)):
        V.append(np.array([np.log(1+rn[i]),(rn[i])**(1/2),1]))
    V = np.array(V)
    try:
        p = lin.solve(V.T @ V, V.T @ racc[1])
        print(p)
        rn = np.array(rn)
        L = lambda r: p[2] + p[1]*r**(1/2) + p[0]*np.log(1+r) + stat.stdev(racc[1])
        #beta_1(r)
        dr = lambda r: -np.array(np.maximum([ ((1+r)**(A/r))*E**2 -(L(r))**2/r**2 ],0))**(1/2)*np.array(np.maximum([(1-2*np.array(m(r))/r)],0))**(1/2) #Oppenheimer-Volkoff
        beta = np.arctan(L(rn)/(-dr(rn)[0][0]*rn))*180/np.pi
        L = lambda r: p[2] + p[1]*r**(1/2) + p[0]*np.log(1+r)
        #beta_2(r)
        dr = lambda r: -np.array(np.maximum([ ((1+r)**(A/r))*E**2 -(L(r))**2/r**2 ],0))**(1/2)*np.array(np.maximum([(1-2*np.array(m(r))/r)],0))**(1/2) #Oppenheimer-Volkoff
        beta -= np.arctan(L(rn)/(-dr(rn)[0][0]*rn))*180/np.pi
        beta = np.abs(beta)
        pfluct.plot(rn,beta,"r")
    except:
        print("no interferenza")
    plt.savefig("C:/Users/raf/Documents/Programmi/tesi/generalizzata/plot" + str(int(xi*(180/np.pi))) + "_fluct.png")