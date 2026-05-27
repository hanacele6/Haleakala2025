pro ptn2atm_new

day='20140511'
filef0='../1.trace/'
filef='../7.subsurf/'
filefp='../7.subsurf+1pm/'
filefm='../7.subsurf-1pm/'
is=10001
ie=10009
file=ie-is+1
iim=161
dw=10

AU=0.3276636178d0		;rms
;gammaD2=0.22426d0       ;@589.1488005nm
;gammaD1=0.2102d0       ;@589.7478099nm
time=dblarr(file)
naatm=dblarr(file)
naerr=dblarr(file)
pi=atan(1d0)*4d0

openr,lunr0,'D:\data\0.2.gamma_factor\gamma_'+day+'.txt',/get_lun
readf,lunr0,wl0,gamma0
gamma=gamma0
free_lun,lunr0
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

openw,lunw,'Na_atoms2.dat',/get_lun
for i=is,ie do begin
  wl=dblarr(iim)
  cts=dblarr(iim)
  ctsp=cts
  ctsm=cts
  sol=cts
  openr,lunr,filef+strtrim(i,2)+'exos.txt',/get_lun
  openr,lunr2,filefp+strtrim(i,2)+'exos.txt',/get_lun	;!!
  openr,lunr3,filefm+strtrim(i,2)+'exos.txt',/get_lun	;!!
  for ii=0,iim-1 do begin
    readf,lunr,wl0,cts0
    wl[ii]=wl0
    cts[ii]=cts0
    readf,lunr2,wl0,err0
    ctsp[ii]=err0
    readf,lunr3,wl0,err0
    ctsm[ii]=err0
  endfor	;end of ii
  err=cts
  free_lun,lunr,lunr2,lunr3

  g=gaussfit(dindgen(2*dw+1),cts[iim/2-dw:iim/2+dw],a,nterms=5)
  c=a[1]+iim/2-dw
  cts2=total(cts[c-3*abs(a[2]):c+3*abs(a[2])])
  width=1+2*3*abs(a[2])
  print,width

  g=gaussfit(dindgen(2*dw+1),ctsp[iim/2-dw:iim/2+dw],a,nterms=5)
  c=a[1]+iim/2-dw
  ctsp2=total(ctsp[c-3*abs(a[2]):c+3*abs(a[2])])
  widthp=1+2*3*abs(a[2])
  print,widthp

  g=gaussfit(dindgen(2*dw+1),ctsm[iim/2-dw:iim/2+dw],a,nterms=5)
  c=a[1]+iim/2-dw
  ctsm2=total(ctsm[c-3*abs(a[2]):c+3*abs(a[2])])
  widthm=1+2*3*abs(a[2])
  print,widthm

;;;;;;Error assessment by STDDEV of wing of Na line
  e1=0d0
  e2=0d0
  for j=0,60-1 do begin
    e1[0]+=err[j]*err[j]
  endfor
  for k=100,160-1 do begin
    e2[0]+=err[k]*err[k]
  endfor
  sigma=sqrt((e1+e2)/119d0)
  err=sigma*SQRT(width)

;;;;;;Error assessment by Fusegawa method
	errf=0d0
	errp=abs(ctsp2-cts2)
	errm=abs(ctsm2-cts2)
	if(errp ge errm)then begin
		errf=errp
	endif else begin
		errf=errm
	endelse

;;;;;Parameter for g-factor
	NaD1=589.7558d0
	c=299792.458d0       ;km/s
	me=9.1093897d-31*1d+3
	e=1.60217733d-19*2.99792458d+8*10d0
	JL=5.18d+14          ;solar flux@1AU [phs/cm2/nm/s]
	f1=0.327d0			 ;f2=0.654d0
;;;;;Calculate number of Na atoms
JLnu1=JL*(1d+9)*((NaD1*1d-9)^2/(c*1d+3))
sigmaD1dnu=!pi*e*e/me/(c*1d+5)*f1	;

gfac1=sigmaD1dnu*JLnu1/AU/AU*gamma
naatm[i-is]=cts2/gfac1;*1d+12
;denserr[i-is]=err/gfac1;*1d+12

naerr[i-is]=(err+errf)/gfac1

printf,lunw,i-10000,naatm[i-is],naerr[i-is]
endfor	;end of i
print,gfac1,err,errf
free_lun,lunw
print,'end'
end