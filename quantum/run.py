import qutip as qt

def solve(Ham, initial_state, t, c_ops, obs_list, args, store_state=True, open=True):
  if open:
    sol = qt.mesolve(Ham, initial_state, t, c_ops=c_ops, e_ops=obs_list, args=args, options={'store_states': store_state})
  else:
    sol = qt.sesolve(Ham, initial_state, t,  e_ops=obs_list, args=args, options={'store_states': store_state})
  return sol