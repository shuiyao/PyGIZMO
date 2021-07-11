import ioformat
import matplotlib.pyplot as plt
from numpy import histogram, logspace, linspace, array, sqrt, log10
from matplotlib.patches import Circle, Rectangle

# Compare SMF at different redshift.
# Write smf_###.txt file

FATHERDIR = "/proj/shuiyao/"
DATABASEDIR = "/proj/shuiyao/"
modelname = "l25n288-phew-m5-spl"
SKIDBASE = modelname+"/"
outfolder = "LoS/"
galfile = DATABASEDIR+SKIDBASE+"gal_z098.stat"
sofile = DATABASEDIR+SKIDBASE+"so_z098.sovcirc"

MASSBIN=12
NHALO=75 # Default is 250, for test use 75

#unit_l = 35714.2857143
#unit_m = 619567.645668
unit_l = 25000.
unit_m = 433697.735404
unit_v = 863.735373678
# unit_l = 50000.
# unit_m = 3469578.81574
# unit_v = 1727.47074736

mstar, mvir, msub = [], [], []
gid = []

fgal = open(galfile, "r")
fso = open(sofile, "r")

x, y, z = [], [], []
# x1, y1, z1 = [], [], []

for line in fgal:
    spt = line.split()
    gid.append(int(spt[0]))
    mstar.append(float(spt[4])*unit_m*1.e10/0.7)
    x.append(float(spt[18]))
    y.append(float(spt[19]))
    z.append(float(spt[20]))

fso.readline()
for line in fso:
    spt = line.split()
    mvir.append(float(spt[1])/0.7)
    msub.append(float(spt[6])/0.7)

fgal.close()
fso.close()

idx_mbin1, idx_mbin2, idx_mbin3 = [], [], []
for i in range(len(mstar)):
    m = log10(msub[i])
    ms = log10(mstar[i])
    if 10.75 < m < 11.25 and 8.3 < ms < 9.3:
        idx_mbin1.append(i)
    elif 11.75 < m < 12.25 and 9.7 < ms < 10.75:
        idx_mbin2.append(i)
    elif 12.75 < m < 13.25:
        idx_mbin3.append(i)

dec1 = len(idx_mbin1) / NHALO
dec2 = len(idx_mbin2) / NHALO
if(dec1 == 0): dec1 = 1
if(dec2 == 0): dec2 = 1

# NOW COMPILE LINES FOR 10e11 HALOS
idxlist, gidlist = [], []
mhlist, mslist = [], []
xlist, ylist, zlist = [], [], []
for i in range(NHALO):
    if MASSBIN == 11:
        idx = idx_mbin1[i*dec1]
    if MASSBIN == 12:
        idx = idx_mbin2[i*dec2]
    idxlist.append(idx)
    gidlist.append(gid[idx])
    mhlist.append(str(log10(msub[idx]))[:5])
    mslist.append(str(log10(mstar[idx]))[:5])
    xlist.append(x[idx])
    ylist.append(y[idx])
    zlist.append(z[idx])
print "Number of Galaxies:", len(idxlist)

def write_file(mh=MASSBIN):
    blist = range(10,300,40) # INTEGERS
    dirstr = ["x+","x-","y+","y-"]
    dx = [1., -1., 0., 0.]
    dy = [0., 0., 1., -1.]
    wrap = 0
    fout = open(outfolder+"loshalo."+modelname+".x.lst", "w")
    fout.write(outfolder+"LOSHalo_"+modelname+"_MH"+str(mh)+"_x+\n")
    fout.write(outfolder+"LOSHalo_"+modelname+"_MH"+str(mh)+"_x-")    
    fout.close()
    fout = open(outfolder+"loshalo."+modelname+"y.lst", "w")
    fout.write(outfolder+"LOSHalo_"+modelname+"_MH"+str(mh)+"_y+\n")
    fout.write(outfolder+"LOSHalo_"+modelname+"_MH"+str(mh)+"_y-")    
    fout.close()
    for j in range(4):
        outname = outfolder+"LOSHalo_"+modelname+"_MH"+str(mh)+"_"+dirstr[j]
        fout = open(outname, "w")
        for b in blist:
            for i in range(NHALO):
                suffix = "."+dirstr[j]+str(b)+"kpc"
                specoutname = str(gidlist[i])+"_"+mhlist[i]+"_"+mslist[i]+suffix
                xcoord = xlist[i]+dx[j]*b/unit_l
                ycoord = ylist[i]+dy[j]*b/unit_l
                if xcoord < -0.5: xcoord += 1.; wrap+=1
                if xcoord > 0.5: xcoord -= 1.; wrap+=1
                if ycoord < -0.5: ycoord += 1.; wrap+=1
                if ycoord > 0.5: ycoord -= 1.; wrap+=1
                line = "% .6f % .6f % .6f 2 " % \
                       (xcoord, ycoord, zlist[i]) + \
                       specoutname+"\n"
                fout.write(line)
        fout.close()
        print "Number of Wrap-ups: ", wrap

x = array(x)
y = array(y)
plt.plot(x[idx_mbin1][::dec1], y[idx_mbin1][::dec1], "b.", alpha=0.5)
plt.plot(x[idx_mbin2][::dec2], y[idx_mbin2][::dec2], "r.", alpha=0.5)
xc = x[idx_mbin2][::dec2]
yc = y[idx_mbin2][::dec2]
for i in range(len(xc)):
    circles = Circle((xc[i], yc[i]), radius=0.006, color="red", alpha=0.2)
    plt.gca().add_artist(circles)
plt.gca().add_artist(Rectangle((-0.5, -0.5), 1., 1., ec="black", fc="none"))
