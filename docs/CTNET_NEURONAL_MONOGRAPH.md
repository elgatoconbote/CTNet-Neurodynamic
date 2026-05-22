# CTNet Neuronal: monografía técnica

## 1. Tesis

CTNet Neuronal no es una red que imita neuronas. Es una arquitectura de estado distribuido en la que el contexto se convierte en geometría dinámica. El sistema no guarda una lista creciente de memoria; deforma un soporte fijo.

```math
CTNet = Core_rev + M_fixed + R_fixed + T_coh + (u,p) + (rho,pi) + Adm + Cards + P_out
```

La salida no agota el estado:

```math
Y_t = P_out(Omega_{t+1}),     Y_t != Omega_{t+1}
```

## 2. Estado

```math
Omega_t = (Z_t, M_t, R_t, C_t, rho_t, pi_t)
```

`Z` es estado distribuido, `M` memoria topológica fija, `R` relaciones reificadas, `rho/pi` régimen espectral-categórico.

## 3. Reversibilidad cardinal

```math
||F^-1(F(Z)) - Z|| < eps_rev
```

El readout no participa en la inversa. La identidad del sistema está antes de la salida.

## 4. Partición u/p

```math
Z_t = [u_t, p_t]
```

`u` despliega, empuja, explora. `p` conserva, arrastra, amortigua. La zona viva exige:

```math
0 < ||u_t|| / ||p_t|| < infinity
```

## 5. Memoria topológica fija

```math
shape(M_{t+1}) = shape(M_t)
M_{t+1} != append(M_t, m_new)
```

La memoria crece por orografía, no por longitud.

## 6. Caos determinista

El residuo caótico es cofibra activa:

```math
nu_chaos,t = Z_t - R_obs O_obs(Z_t)
```

Funciona si:

```math
lambda_max > 0  and  C_t > C_min  and  D_inc < D_max
```

El caos no se suprime. Se disciplina por `u/p + T_coh`.

## 7. Tensor de coherencia

```math
T_coh = T(I/d, H, E_coh, phi, varsigma, D_inc, M, R, A)
```

Convierte residuo caótico en masa, velocidad, admisibilidad y cierre:

```math
T_coh : nu_chaos -> (G, C, mu, Adm, s)
```

## 8. Régimen ejecutivo

```math
pi_t = softmax(W_pi z_t + b_pi)
rho_t = argmax_i pi_{t,i}
```

Fases canónicas:

```text
explore, decide, reflect, stabilize, verify, project
```

Esto es el espectro categórico: continuidad interna, categoría operativa.

## 9. Shock e individuación

Un individuo no es un ID. Es trayectoria plegada:

```math
I_a(t) = (M_a, R_a, u_a, p_a, rho_a, pi_a, T_coh,a, D_inc,a)
```

Respuesta al shock:

```math
ShockResponse_i(x) = F(x, M_i, R_i, u_i, p_i, rho_i, pi_i, T_coh,i, D_inc,i)
```

Dos individuos con misma arquitectura pueden responder distinto porque su topología interna divergió históricamente.

## 10. Ecuación maestra

```math
M_t = U_M(Z_t, M_{t-1}, C_t)
rho_t = U_rho(Z_t, M_t, e_task)
Zbar_{t+1} = F_theta(Z_t)
A_t = A(Zbar_{t+1}, rho_t)
C_t = C(Zbar_{t+1}, M_t, R_t, A_t)
Delta Z_t = Delta M_t + Delta rho_t + Delta A_t + Delta C_t
Z_{t+1} = Norm(Zbar_{t+1} + s_t Delta Z_t)
R_{t+1} = U_R(R_t, Z_{t+1}, M_{t+1})
Y_t = P_out(Z_{t+1}, M_{t+1}, R_{t+1})
```

## 11. Auditoría

CTNet se mantiene si:

```text
core reversible
memory fixed shape
relations fixed shape
no append
readout separate
coherence affects transition
regime affects transition
speed bounded
multi-card readout remains projective
```

Regla de oro:

```text
If it requires growing memory, output as identity, or a broken reversible core,
it is not yet in CTNet form.
```
