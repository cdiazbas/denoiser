import astropy.io.fits as fits

def writefits(name, d):
    io = fits.PrimaryHDU(d)
    io.writeto(name, overwrite=True)

def readfits(name):
    return fits.getdata(name, ext=0)
