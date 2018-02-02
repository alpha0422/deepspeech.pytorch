import torch
from torch.autograd import Variable
from torch.autograd import Function

import time
import pdb

# Keep global layer id
layer_ids = set()
count = 1

class Profiling(object):
    def __init__(self, model):
        if isinstance(model, torch.nn.Module) is False:
            print("Not a valid model, please provide a 'nn.Module' instance.")

        self.model = model
        self.record = {'forward':[], 'backward': []}
        self.profiling_on = True
        self.origin_call = {}
        self.hook_done = False
        self.layer_num = 0

    def __enter__(self):
        self.start()

        return self

    def __exit__(self, *args):
        self.stop()

    def __str__(self):
        ret = ""

        iter = len(self.record['forward']) / self.layer_num

        for i in xrange(iter):
            ret += "\n================================= Iteration {} =================================\n".format(i + 1)

            ret += "\nFORWARD TIME:\n\n"
            for j in xrange(self.layer_num):
                record_item = self.record['forward'][i * self.layer_num + j]
                ret += "layer{:3d}:          {:.6f} ms          ({})\n".format(j + 1, record_item[2] - record_item[1], record_item[0])

            ret += "\nBACKWARD TIME:\n\n"
            for j in (xrange(self.layer_num)):
                record_item = self.record['backward'][i * self.layer_num + self.layer_num - j - 1]
                try:
                    ret += "layer{:3d}:          {:.6f} ms          ({})\n".format(j + 1, record_item[2] - record_item[1], record_item[0])
                except:
                    # Oops, this layer doesn't execute backward post-hooks
                    pass

        return ret

    def start(self):
        if self.hook_done is False:
            self.hook_done = True
            self.hook_modules(self.model)

        self.profiling_on = True

        # Print table header
        print 'Layer,FMA,Input,Weight,Output'

        return self

    def stop(self):
        self.profiling_on = False

        return self

    def hook_modules(self, module):
        this_profiler = self
        sub_modules = module._modules

        for name, sub_module in sub_modules.items():
            if len(sub_module._modules) > 0:
                self.hook_modules(sub_module)
            else:
                # nn.Module who doesn't have sub nn.Module, hook it.
                self.layer_num += 1

                # Wrapper function to "__call__", with time counter in it.
                def wrapper_call(self, *input, **kwargs):
                    global count

                    # Push nvtx range and sol calculator
                    torch.cuda.nvtx.range_push('Fprop ' + str(self).split('(')[0].strip()
                            +'.{}'.format(count))

                    # Call the origin function
                    start_time = time.time()
                    results = this_profiler.origin_call[self.__class__](self, *input, **kwargs)
                    result = results[0] if type(results) is tuple else results
                    stop_time = time.time()

                    # Pop nvtx range
                    torch.cuda.nvtx.range_pop()
                    that = self

                    # Print FMA and Input/Weight/Output
                    if id(self) not in layer_ids and not isinstance(self, torch.nn.Hardtanh):
                        # Avoid redundant
                        layer_ids.add(id(self))

                        # Prepare the statistics
                        layer_name = str(self).split('(')[0].strip() + '.{}'.format(count)
                        assert len(input) == 1
                        func = lambda op, size: op.join(map(str, size))
                        layer_input = func('*', input[0].size())
                        layer_weight = func('+', map(lambda param: func('*', param.size()), self.parameters()))
                        layer_output = func('*', result.size())

                        try:
                            # Calculate FMA
                            if isinstance(self, torch.nn.Conv1d):
                                # Assume group = hidden
                                l,n,g = result.size()
                                r, = self.kernel_size
                                layer_fma = '{}*{}*{}*{}'.format(l,r,g,n)
                            elif isinstance(self, torch.nn.Conv2d):
                                n,k,h,w = result.size()
                                c = input[0].size()[1]
                                r,s = self.kernel_size
                                layer_fma = '({N}*{H}*{W})*({C}*{R}*{S})*({K})'.format(N=n,H=h,W=w,C=c,R=r,S=s,K=k)
                            elif isinstance(self, torch.nn.GRU):
                                l,n,_ = input[0].size()
                                i2h, h2h = map(lambda param: func('*', param.size()), self.parameters())
                                layer_fma = '{i2h}*({L}*{N})+{h2h}*({L}*{N})'.format(i2h=i2h,h2h=h2h,L=l,N=n)
                            elif isinstance(self, torch.nn.Linear):
                                wh, ww = self.out_features, self.in_features
                                layer_fma = '{}*{}*{}'.format(wh,ww,input[0].size()[0])
                            else:
                                layer_fma = ''

                            # Print the data for SOL
                            print layer_name+','+','.join(map(lambda x: x if x is '' else '={}'.format(x), (layer_fma, layer_input, layer_weight, layer_output)))
                        except:
                            print layer_name

                    # Update layer count
                    count = count + 1

                    def backward_pre_hook(*args):
                        if (this_profiler.profiling_on):
                            global count
                            count = count - 1
                            torch.cuda.nvtx.range_push('Bprop ' + str(self).split('(')[0].strip()
                                    +'.{}'.format(count))
                            this_profiler.record['backward'].append((that, time.time()))

                    result.grad_fn.register_pre_hook(backward_pre_hook);

                    if (this_profiler.profiling_on):
                        global record
                        this_profiler.record['forward'].append((self, start_time, stop_time))

                    return results

                # Replace "__call__" with "wrapper_call".
                if sub_module.__class__ not in this_profiler.origin_call:
                    this_profiler.origin_call.update({sub_module.__class__: sub_module.__class__.__call__})
                    sub_module.__class__.__call__ = wrapper_call

                def backward_post_hook(*args):
                    if (this_profiler.profiling_on):
                        this_profiler.record['backward'][-1] = (this_profiler.record['backward'][-1][0], this_profiler.record['backward'][-1][1], time.time())
                        torch.cuda.nvtx.range_pop()

                sub_module.register_backward_hook(backward_post_hook)
