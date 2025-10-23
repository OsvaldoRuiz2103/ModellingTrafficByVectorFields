import numpy as np

def field_metrics(u, v, grid: float, y_up=True):
    vy = -v if y_up else v
    h = float(grid)

    du_dy, du_dx = np.gradient(u, h, h)   
    dv_dy, dv_dx = np.gradient(vy, h, h)

    div  = du_dx + dv_dy
    curl = dv_dx - du_dy                 

    lap_u = np.gradient(du_dx, h, axis=1) + np.gradient(du_dy, h, axis=0)
    lap_v = np.gradient(dv_dx, h, axis=1) + np.gradient(dv_dy, h, axis=0)

    speed = np.hypot(u, vy)
    K = 0.5 * speed**2

    s_n = du_dx - dv_dy
    s_s = du_dy + dv_dx
    W   = s_n**2 + s_s**2 - curl**2
    Q   = 0.5*(curl**2 - (s_n**2 + s_s**2))

    return dict(div=div, curl=curl, lap_u=lap_u, lap_v=lap_v,
                speed=speed, K=K, s_n=s_n, s_s=s_s, W=W, Q=Q)
