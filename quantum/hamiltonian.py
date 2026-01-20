import numpy as np

def g_t(t, args):
    if args['coupling'] == 'cos':
        coupling = args['g0'] * np.cos(args['w'] * t + args['phi'])
    elif args['coupling'] == 'sin':
        coupling =  args['g0'] * np.sin(args['w'] * t)
    elif args['coupling'] == 'linear':
        coupling = args['g0'] *( args['w'] * t + args['phi'])
    elif args['coupling'] == 'exp':
        coupling = args['g0']* np.exp(args['w'] * t + args['phi'])
    elif args['coupling'] == 'gauss' and 'sigma' in args:
        coupling = args['g0'] * np.exp(args['sigma']*((t - args['T'])/args['epsilon'])**2)
    return coupling

def h_closed(args, b, sp, sm):
    return args['g0']*(sp * b + sm * b.dag())

def h_open(b, sp, sm):
    return [[(sp * b + sm * b.dag()), g_t]]