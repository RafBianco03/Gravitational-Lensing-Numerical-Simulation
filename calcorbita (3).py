import numpy as np
import scipy.interpolate as interp
import scipy.optimize as op
import scipy.integrate as integ
import matplotlib.pyplot as plt
import numpy.linalg as lin
import scipy.interpolate as interp

#Costanti del moto
xi = 0
L = 0
E = 1
A = 10**6

#Numero di punti generati per la rappresentazione di ogni traiettoria luce
prec = int(1e4)

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
r_min1 = op.fsolve(eq, 1)[0]
print(r_min1)
OC = 0

def r(a,b,d): #Voglio calcolare r(φ), avendo dr/dφ
    ang1 = np.linspace(a,b,prec) #Prendo un linspace di angoli arbitrario
    #Metodo solve_ivp
    ang_span = [a,b] #Intervallo di angoli
    sol = integ.solve_ivp(dr_dphi, ang_span, [d], dense_output=True) #Genera la soluzione all'equazione differenziale, sugli angoli ang_span e un raggio iniziale r(a)=d
    r = sol.sol(ang1)[0] #Genero la soluzione sullo span

    #if (((1+r)**(A/r))*E**2 -(L)**2/r**2 < 0).any():
    #    return None
    if ((1-2*np.array(m(r))/r) < 0).any():
        print("Fuori dalle Ipotesi")
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

def campoVettoriale(r,ang,L,att):
    prima = dr_(r)
    seconda = L/r**2
    if att:
        return r*np.cos(ang),r*np.sin(ang),prima*np.cos(ang)-r*np.sin(ang)*seconda,np.sin(ang)*prima+r*np.cos(ang)*seconda
    else:
        return r*np.cos(ang),r*np.sin(ang),-prima*np.cos(ang)-r*np.sin(ang)*seconda,np.sin(ang)*(-prima)+r*np.cos(ang)*seconda

def star(An,E): #Luce proveniente da una sola sorgente che emette isotropicamente
    global L
    global OC
    global A
    A = An
    
    n = 500 #Numero di fotoni
    rs = [] #Raggio vettore delle soluzioni
    ang = [] #Angoli delle soluzioni

    angolo = np.pi #Scelgo l'angolo iniziale
    dist_max = 10**10

    alfa = [] #Angoli di intercetta tra la geodetica luce e la linea di vista (L.O.S.)
    racc = [[],[]] #Informazioni di raggio di intercetta e momento angolare L

    L_max = E*dist_max*np.e**(-pot(dist_max))
    Ls = np.linspace(0.001,L_max,n) #Creo uno span di momenti angolari tra due estremi

    for i in range(len(Ls)): #Plot di due parti simmetriche rispetto all'asse x corrispondenti a momenti angolari positivi e negativi
        L = Ls[i] #Assegno il momento angolare per il ciclo
        try:
            a,b = r(-angolo, np.pi-angolo,dist_max) #Span di angoli

            #Multiimmagini a raggio:
            interferenza_b = np.abs(-b+xi) < 0.001 #Quando l'argomento angolare è nullo, ci troviamo sulla linea di vista L.O.S.

            #Campo vettoriale quadrivettore K
            prima = -dr(a)[0][0]
            seconda = L/a**2
            #Angolo d'inclinazione del vettore d'onda al momento dell'intercetta
            #phit = np.arctan((np.sin(b[interferenza])*prima[interferenza]+a[interferenza]*np.cos(b[interferenza])*seconda[interferenza])/(prima[interferenza]*np.cos(b[interferenza])-a[interferenza]*np.sin(b[interferenza])*seconda[interferenza]))
            phit_b = np.arctan((a[interferenza_b]*seconda[interferenza_b])/(prima[interferenza_b]))

            L *= -1 #Plot identico speculare all'asse x
            c,d = r(-angolo, np.pi-angolo,dist_max)
            #Multiimmagini a raggio:
            interferenza_d = np.abs(-d+xi) < 0.001 #Quando l'argomento angolare è nullo, ci troviamo sulla linea di vista L.O.S.

            phit_d = np.arctan((a[interferenza_d]*seconda[interferenza_d])/(prima[interferenza_d]))
            
            #Raccolgo le informazioni dell'onda i-esima
            if(len(a[interferenza_b])!=0 and len(c[interferenza_d] != 0)):
                alfa.append((phit_d[0]+phit_b[0])*180/np.pi)
                racc[0].append(a[interferenza_d][0])
                racc[1].append(-L)
                OC += 1

            rs.append(a)
            rs.append(c)
            ang.append(b)
            ang.append(d)
        except:
            continue
    return rs,ang,racc,alfa

xi = 0

fig, ax = plt.subplots(2,1)
plot = ax[0]
stella = star(A,E)
rs,ang,racc,alfa = stella[0],stella[1],stella[2],stella[3]

#print(OC/len(rs))

#Ora abbiamo (t,r,θ,φ), rappresentiamo l'orbita in 2D
for i in range(len(rs)):
    x = rs[i]*np.cos(ang[i])
    y = rs[i]*np.sin(ang[i])
    plot.plot(x,y)
plot.grid()
plot.axis("equal")
plot.axhline(0, color='black', linewidth=1, ls='--')  # Linea orizzontale a y=0
plot.axvline(0, color='black', linewidth=1, ls='--')  # Linea verticale a x=0
raggio_x = []
raggio_y = []
xs = np.linspace(0,10,100)
for i in range(len(xs)):
    raggio_x.append(xs[i]*np.cos(xi))
    raggio_y.append(xs[i]*np.sin(xi))
#plot.plot(np.array(raggio_x),np.array(raggio_y),"red")
plot.set_xlabel("r/$r_s$")
plot.set_ylabel("r/$r_s$")

plotAnello = ax[1]

#Ordino i dati
rn = sorted(racc[0])
ln = []
alpha = []
for i in range(len(rn)):
    ln.append(racc[1][racc[0].index(rn[i])])
    alpha.append(alfa[racc[0].index(rn[i])])


#Plot del diametro dell'anello di Einstein al raggio L.O.S.E.(line of sight event)
plotAnello.set_ylabel("Diametro angolare (°)")
plotAnello.set_xlabel("Distanza dalla lente r ($r_{s}$ unit)")
plotAnello.plot(rn,alpha,"o")
plt.show()


#Algoritmi per la stima di L(r) in funzione di E ed A
def L_A(A):
    L_ = []
    for j in range(len(A)):
        try:
            Ln = [0,0,0]
            print("A:" + str(A[j]) + "E:" + str(E))
            racc = star(A[j],E)[2]
            #Ordino i dati
            rn = sorted(racc[0])
            ln = []
            for i in range(len(rn)):
                ln.append(racc[1][racc[0].index(rn[i])])

            #Approssima L(r) con funzione lineare nei parametri
            V = []
            for i in range(len(rn)):
                V.append(np.array([np.log(1+rn[i]),(rn[i])**(1/2),1]))
            V = np.array(V)
            try:
                p = lin.solve(V.T @ V, V.T @ ln)
                for i in range(len(Ln)):
                    Ln[i] = p[i]
            except:
                for i in range(len(Ln)):
                   Ln[i].append(0)
            L_.append(Ln)
        except:
            L_.append(np.zeros(3))
        #Ogni giro inserisce i coefficienti a,b,c per le scelte (A,E) nell'array Ln, infine L_ è l'insieme degli Ln
    return L_

"""
#Plotta L(r) con A e E iniziali
plt.plot(rn,ln,"o")
p = L_A([A])[0]
print(p)
xn = np.linspace(min(rn),max(rn),500)

L = lambda r: p[2] + p[1]*r**(1/2) + p[0]*np.log(1+r)
xn = np.delete(xn,L(xn)<0)
ynn = L(xn)

plt.xlabel("Distanza dalla lente ($r_{s}$ unit)")
plt.ylabel("Momento angolare $L(r)$")
plt.plot(xn,ynn)
plt.show()

#Plotta alfa_E(r)
dr = lambda r: -np.array(np.maximum([ ((1+r)**(A/r))*E**2 -(L(r))**2/r**2 ],0))**(1/2)*np.array(np.maximum([(1-2*np.array(m(r))/r)],0))**(1/2) #Oppenheimer-Volkoff
alpha_E = lambda r: 2*np.arctan(L(r)/(-dr(r)[0][0]*r))*180/np.pi
rn = np.array(rn)
plt.plot(rn, alpha, "o")
xn = np.delete(xn, alpha_E(xn)<0)
plt.plot(xn, alpha_E(xn))
plt.ylabel("Diametro angolare $alpha_E(r)$(°)")
plt.xlabel("Distanza dalla lente ($r_{s}$ unit)")
plt.show()
"""

#Plotta i parametri in funzione di A
An = np.linspace(0,5,100)
#An = np.array([0.01,1])

cost = []
log = []
rad = []

try:
    a = open("a.txt", "rb")
    b = open("b.txt", "rb")
    c = open("c.txt", "rb")
    cost = np.load(a)
    log = np.load(b)
    rad = np.load(c)
    a.close()
    b.close()
    c.close()
except:
    L_ = np.array(L_A(An))
    for i in range(len(L_)):
        #Qui sto scorrendo sulla scelta di A
        cost.append(L_[i][2]) #Qui ho i valori della parametro costante per ogni scelta (A,E)
        rad.append(L_[i][1])
        log.append(L_[i][0])

fig, ax = plt.subplots(3,1)

ax[0].set_xlabel("A")
ax[1].set_xlabel("A")
ax[2].set_xlabel("A")
ax[0].set_ylabel("c")
ax[1].set_ylabel("b")
ax[2].set_ylabel("a")
ax[0].plot(An, rad, "o")
ax[1].plot(An, log, "o")
ax[2].plot(An, cost, "o")

a = open("a.txt", "wb")
b = open("b.txt", "wb")
c = open("c.txt", "wb")
np.save(a, cost)
np.save(b, log)
np.save(c, rad)
a.close()
b.close()
c.close()

#Sembrano esponenziali a(E,A)/E, ecc.
V = []
for i in range(len(An)):
    V.append([np.e**An[i]**2])
V = np.array(V)
print(V.T@V)
cost = np.array(cost)
log = np.array(log)
rad = np.array(rad)
pcost = lin.solve(V.T @ V, V.T @ cost)
prad = lin.solve(V.T @ V, V.T @ rad)
plog = lin.solve(V.T @ V, V.T @ log)
ax[2].plot(An, pcost[0]*np.e**An**2)
ax[0].plot(An, prad[0]*np.e**An**2)
ax[1].plot(An, plog[0]*np.e**An**2)
print(pcost)
print(prad)
print(plog)
plt.show()
