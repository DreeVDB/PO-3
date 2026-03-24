import casadi as ca

x = ca.SX.sym('x')
y = ca.SX.sym('y')
qp = {'x': ca.vertcat(x, y), 'f': x**2 + y**2, 'g': x + y - 10}

S = ca.qpsol('S', 'qpoases', qp)
r = S(lbg=0)
print(r['x'])