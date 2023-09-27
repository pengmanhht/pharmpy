$PROBLEM DIRECT_EMAX
$DATA ../.datasets/input_model.csv IGNORE=@
$INPUT ID TIME AMT WGT APGR DV DVID
$SUBROUTINE ADVAN1 TRANS2

$ABBR REPLACE ETA_E_MAX=ETA(3)
$ABBR REPLACE ETA_B=ETA(4)
$PK
EC_50 = THETA(5)
E_MAX = THETA(4)*(ETA_E_MAX + 1)
B = THETA(3)*EXP(ETA_B)
CL = THETA(1)*EXP(ETA(1))
VC = THETA(2)*EXP(ETA(2))
V=VC

$ERROR
CONC = A(1)/VC
Y = CONC + CONC*EPS(1)
E = B*(A(1)*E_MAX/(V*(A(1)/V + EC_50)) + 1)
Y_2 = E + E*EPS(2)

IF (DVID.EQ.1) THEN
    Y = Y
ELSE
    Y = Y_2
END IF
$THETA (0,0.00273872) FIX ; TVCL
$THETA (0,1.44718) FIX ; TVV
$THETA  (0,5.75005) ; POP_B
$THETA  (-1,0.1) ; POP_E_MAX
$THETA  (0,0.1) ; POP_EC_50
$OMEGA 3.09626e-06 FIX  ; IVCL
$OMEGA 3.1128e-06 FIX  ; IVV
$OMEGA  0.09 ; IIV_E_MAX
$OMEGA  0.09 ; IIV_B
$SIGMA 0.00630754 FIX

$SIGMA  0.338363 ; sigma
$ESTIMATION METHOD=1 INTERACTION

$TABLE ID TIME CWRES NOAPPEND
       NOPRINT ONEHEADER FILE=pd_sdtab1
