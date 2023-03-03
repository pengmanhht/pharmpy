import sympy

from pharmpy.modeling import (
    add_peripheral_compartment,
    remove_peripheral_compartment,
    set_peripheral_compartments,
)


def test_advan1(create_model_for_test):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V=VC
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = create_model_for_test(code, dataset='pheno')
    model = add_peripheral_compartment(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN3 TRANS4
$PK
VP1 = THETA(4)
QP1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V1 = VC
Q = QP1
V2 = VP1
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA  (0,0.00469307) ; POP_QP1
$THETA  (0,0.011000000000000001) ; POP_VP1
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct
    odes = model.statements.ode_system
    central = odes.central_compartment
    periph = odes.peripheral_compartments[0]
    rate = model.statements.ode_system.get_flow(central, periph)
    assert rate == sympy.Symbol('Q') / sympy.Symbol('V1')


def test_advan2(create_model_for_test):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN2 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
MAT=THETA(3)*EXP(ETA(3))
V=VC
KA=1/MAT
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$OMEGA 0.0309626  ; IVMAT
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = create_model_for_test(code, dataset='pheno')
    model = add_peripheral_compartment(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN4 TRANS4
$PK
VP1 = THETA(4)
QP1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
MAT=THETA(3)*EXP(ETA(3))
V2 = VC
KA=1/MAT
Q = QP1
V3 = VP1
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA  (0,0.00469307) ; POP_QP1
$THETA  (0,0.011000000000000001) ; POP_VP1
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$OMEGA 0.0309626  ; IVMAT
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct


def test_advan2_trans1(create_model_for_test):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN2 TRANS1
$PK
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
MAT=THETA(3)*EXP(ETA(3))
KA=1/MAT
K=CL/VC
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$OMEGA 0.0309626  ; IVMAT
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = create_model_for_test(code, dataset='pheno')
    model = add_peripheral_compartment(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN4 TRANS1
$PK
VP1 = THETA(4)
QP1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
MAT=THETA(3)*EXP(ETA(3))
KA=1/MAT
K23 = QP1/VC
K32 = QP1/VP1
K=CL/VC
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA  (0,0.00469307) ; POP_QP1
$THETA  (0,0.011000000000000001) ; POP_VP1
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$OMEGA 0.0309626  ; IVMAT
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct


def test_advan3(create_model_for_test):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN3 TRANS4
$PK
VP1 = THETA(4)
QP1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V1 = VC
Q = QP1
V2 = VP1
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA  (0,0.1) ; POP_QP1
$THETA  (0,0.1) ; POP_VP1
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = create_model_for_test(code, dataset='pheno')
    model = add_peripheral_compartment(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN11 TRANS4
$PK
VP2 = THETA(6)
QP2 = THETA(5)
VP1 = THETA(4)
QP1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V1 = VC
Q2 = QP1
V2 = VP1
Q3 = QP2
V3 = VP2
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA  (0,0.010000000000000002) ; POP_QP1
$THETA  (0,0.1) ; POP_VP1
$THETA  (0,0.09000000000000001) ; POP_QP2
$THETA  (0,0.1) ; POP_VP2
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct


def test_advan4(create_model_for_test):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN4 TRANS4
$PK
VP1 = THETA(4)
QP1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
MAT=THETA(3)*EXP(ETA(3))
V2 = VC
KA=1/MAT
Q = QP1
V3 = VP1
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA  (0,0.1) ; POP_QP1
$THETA  (0,0.1) ; POP_VP1
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$OMEGA 0.0309626  ; IVMAT
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = create_model_for_test(code, dataset='pheno')
    model = add_peripheral_compartment(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN12 TRANS4
$PK
VP2 = THETA(6)
QP2 = THETA(5)
VP1 = THETA(4)
QP1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
MAT=THETA(3)*EXP(ETA(3))
V2 = VC
KA=1/MAT
Q3 = QP1
V3 = VP1
Q4 = QP2
V4 = VP2
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA  (0,0.010000000000000002) ; POP_QP1
$THETA  (0,0.1) ; POP_VP1
$THETA  (0,0.09000000000000001) ; POP_QP2
$THETA  (0,0.1) ; POP_VP2
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$OMEGA 0.0309626  ; IVMAT
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct


def test_advan1_two_periphs(create_model_for_test):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V=VC
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = create_model_for_test(code, dataset='pheno')
    model = add_peripheral_compartment(model)
    model = add_peripheral_compartment(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN11 TRANS4
$PK
VP2 = THETA(6)
QP2 = THETA(5)
VP1 = THETA(4)
QP1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V1 = VC
Q2 = QP1
V2 = VP1
Q3 = QP2
V3 = VP2
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA  (0,0.00046930699999999997) ; POP_QP1
$THETA  (0,0.011000000000000001) ; POP_VP1
$THETA  (0,0.004223763) ; POP_QP2
$THETA  (0,0.011000000000000001) ; POP_VP2
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct


def test_advan1_remove(create_model_for_test):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA run1.csv IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V=VC
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = create_model_for_test(code)
    model = remove_peripheral_compartment(model)
    assert model.model_code == code


def test_advan3_remove(create_model_for_test):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN3 TRANS4
$PK
VP1 = THETA(4)
QP1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V1 = VC
Q = QP1
V2 = VP1
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA  (0,0.1) ; POP_QP1
$THETA  (0,0.1) ; POP_VP1
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = create_model_for_test(code, dataset='pheno')
    model = remove_peripheral_compartment(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V = VC
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,2.350801373088405) ; POP_VC
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct


def test_advan4_remove(create_model_for_test):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN4 TRANS4
$PK
VP1 = THETA(4)
QP1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
MAT=THETA(3)*EXP(ETA(3))
V2 = VC
KA=1/MAT
Q = QP1
V3 = VP1
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA (0,0.1) ; POP_MAT
$THETA  (0,0.1) ; POP_QP1
$THETA  (0,0.1) ; POP_VP1
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$OMEGA 0.0309626  ; IVMAT
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = create_model_for_test(code, dataset='pheno')
    model = remove_peripheral_compartment(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN2 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
MAT=THETA(3)*EXP(ETA(3))
V = VC
KA=1/MAT
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,2.350801373088405) ; POP_VC
$THETA (0,0.1) ; POP_MAT
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$OMEGA 0.0309626  ; IVMAT
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct


def test_advan11_remove(create_model_for_test):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN11 TRANS4
$PK
VP2 = THETA(6)
QP2 = THETA(5)
VP1 = THETA(4)
QP1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V1 = VC
Q2 = QP1
V2 = VP1
Q3 = QP2
V3 = VP2
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA  (0,0.1) ; POP_QP1
$THETA  (0,0.1) ; POP_VP1
$THETA  (0,0.1) ; POP_QP2
$THETA  (0,0.1) ; POP_VP2
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = create_model_for_test(code, dataset='pheno')
    model = remove_peripheral_compartment(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN3 TRANS4
$PK
VP1 = THETA(4)
QP1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V1 = VC
Q = QP1
V2 = VP1
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA  (0,0.1) ; POP_QP1
$THETA  (0,0.2) ; POP_VP1
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct


def test_advan12_remove(create_model_for_test):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN12 TRANS4
$PK
VP2 = THETA(6)
QP2 = THETA(5)
VP1 = THETA(4)
QP1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
MAT=THETA(3)*EXP(ETA(3))
V2 = VC
KA=1/MAT
Q3 = QP1
V3 = VP1
Q4 = QP2
V4 = VP2
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA  (0,0.1) ; POP_QP1
$THETA  (0,0.1) ; POP_VP1
$THETA  (0,0.1) ; POP_QP2
$THETA  (0,0.1) ; POP_VP2
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$OMEGA 0.0309626  ; IVMAT
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = create_model_for_test(code, dataset='pheno')
    model = remove_peripheral_compartment(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN4 TRANS4
$PK
VP1 = THETA(4)
QP1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
MAT=THETA(3)*EXP(ETA(3))
V2 = VC
KA=1/MAT
Q = QP1
V3 = VP1
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA  (0,0.1) ; POP_QP1
$THETA  (0,0.2) ; POP_VP1
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$OMEGA 0.0309626  ; IVMAT
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct


def test_advan11_remove_two_periphs(create_model_for_test):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN11 TRANS4
$PK
VP2 = THETA(6)
QP2 = THETA(5)
VP1 = THETA(4)
QP1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V1 = VC
Q2 = QP1
V2 = VP1
Q3 = QP2
V3 = VP2
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA  (0,0.1) ; POP_QP1
$THETA  (0,0.1) ; POP_VP1
$THETA  (0,0.1) ; POP_QP2
$THETA  (0,0.1) ; POP_VP2
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = create_model_for_test(code, dataset='pheno')
    model = remove_peripheral_compartment(model)
    model = remove_peripheral_compartment(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V = VC
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,4.48160274617681) ; POP_VC
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct


def test_advan4_roundtrip(create_model_for_test):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN4 TRANS4
$PK
VP1 = THETA(5)
QP1 = THETA(4)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
MAT=THETA(3)*EXP(ETA(3))
V2 = VC
KA=1/MAT
Q = QP1
V3 = VP1
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA (0,0.1) ; POP_MAT
$THETA  (0,0.1) ; POP_QP1
$THETA  (0,0.1) ; POP_VP1
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$OMEGA 0.0309626  ; IVMAT
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = create_model_for_test(code, dataset='pheno')
    model = add_peripheral_compartment(model)
    model = remove_peripheral_compartment(model)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN4 TRANS4
$PK
VP1 = THETA(5)
QP1 = THETA(4)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
MAT=THETA(3)*EXP(ETA(3))
V2 = VC
KA=1/MAT
Q = QP1
V3 = VP1
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA (0,0.1) ; POP_MAT
$THETA  (0,0.05) ; POP_QP1
$THETA  (0,0.2) ; POP_VP1
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$OMEGA 0.0309626  ; IVMAT
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct


def test_set_peripheral_compartments(create_model_for_test):
    code = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN11 TRANS4
$PK
VP2 = THETA(6)
QP2 = THETA(5)
VP1 = THETA(4)
QP1 = THETA(3)
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V1 = VC
Q2 = QP1
V2 = VP1
Q3 = QP2
V3 = VP2
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,0.22) ; POP_VC
$THETA  (0,0.1) ; POP_QP1
$THETA  (0,0.1) ; POP_VP1
$THETA  (0,0.1) ; POP_QP2
$THETA  (0,0.1) ; POP_VP2
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    model = create_model_for_test(code, dataset='pheno')
    model = set_peripheral_compartments(model, 0)
    correct = """$PROBLEM PHENOBARB SIMPLE MODEL
$DATA pheno.dta IGNORE=@
$INPUT ID TIME AMT WGT APGR DV FA1 FA2
$SUBROUTINE ADVAN1 TRANS2
$PK
CL=THETA(1)*EXP(ETA(1))
VC=THETA(2)*EXP(ETA(2))
V = VC
$ERROR
Y=F+F*EPS(1)
$THETA (0,0.00469307) ; POP_CL
$THETA (0,4.48160274617681) ; POP_VC
$OMEGA 0.0309626  ; IVCL
$OMEGA 0.0309626  ; IVV
$SIGMA 0.013241
$ESTIMATION METHOD=1 INTERACTION
"""
    assert model.model_code == correct
