import types
import torch
from .sparse_masklib import create_mask

torchvision_imported=True
try:
    import torchvision
except ImportError:
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print("[ASP][Warning] torchvision cannot be imported, may influence functionality of MaskRCNN/KeypointRCNN network from torchvision.")
    torchvision_imported=False

torchvision_ops_supported=True
try:
    import torchvision.ops
except ImportError:
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print("[ASP][Warning] torchvision has no attribute 'ops', may influence functionality of MaskRCNN/KeypointRCNN network from torchvision.")
    torchvision_ops_supported=False

import logging

def eligible_modules(model, whitelist_layer_types, allowed_layer_names, disallowed_layer_names):
    eligible_modules_list = []
    for name, mod in model.named_modules():
        if isinstance(mod, whitelist_layer_types) and name not in disallowed_layer_names:
            if allowed_layer_names is not None and name not in allowed_layer_names:
                continue
            eligible_modules_list.append((name, mod))
    return eligible_modules_list

class ASP:
    __model = None
    __verbosity = 0
    __optimizer = None
    __sparse_parameters = []
    __calculate_mask = None
    __enable_raw_print = True
    __relaxed_tc_check = 0

    @classmethod
    def init_model_for_pruning(cls, model, mask_calculator="m4n2_1d",
             verbosity=3,
             whitelist=[torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d], 
             allowed_layer_names=None, disallowed_layer_names=[],
             allow_recompute_mask=False, custom_layer_dict={},
             enable_raw_print=True, enable_logger_print=False, pass_in_logger=None,
             relaxed_tc_check=0):
        """Call this method to modify your model to take advantage of sparse matrix multiplication.
        Note that this call alone only augments the model with additional buffers needed for sparse MMA,
        it does not enable use of sparse MMA. 

        If you are starting with a fresh model:

        model = ...
        ASP.init_model_for_pruning(model, mask_calculator, ...)
        if (training) ASP.init_optimizer_for_pruning(optimizer)
        #ASP.enable_sparsity() // enable_sparsity function is deprecated.
        ASP.compute_sparse_masks() // sparsity is off by default, call when youy want to enable it.

        If you are starting from a checkpoint:

        model = ...
        ASP.init_model_for_pruning(model, mask_calculator, ...)
        torch.load(...)
        if (training) ASP.init_optimizer_for_pruning(optimizer)

        Arguments:
          model                    The model
          mask_calculator          Either callable that computes mask given a tensor OR pattern string for sparse mask lib.
          verbosity                Integer controling verbosity level.
                                   0 -> Only errors.
                                   1 -> Errors and warnings.
                                   2 -> Errors, warnings and info.
                                   3 -> Errors, warnings, info and debug.
          whitelist                Module types approved for sparsity.
          allowed_layer_names      If not None, only layer names that appear in this list are considered for sparsity.
          disallowed_layer_names   If not [], only layer names that do not appear in this list are considered for sparsity.
          allow_recompute_mask     If True, stores pruned values so that dense weights can be restored.
                                   Pruned weights are stored in CPU memory, hence this option does not increase GPU memory usage.
          custom_layer_dict        Dictionary of additional layer paremeters to sparsify. e.g. {CustomLinear: ['weight']}
                                   # Accept custom (layer type:param name) to include in sparse_parameter dictionary
                                   # Support to include in sparse_parameter_list an user-supplied custom layer type and its parameter name. This is useful when users have their own implementation of nn.Linear or nn.Conv2D. For example, huggingface repo has a custom implementation of nn.Linear called LinearActivation.

          [Future] Support for allow_recompute_mask can be removed, it is not part of sparse inference recipe -- AKM.
          [Chong]
          enable_raw_print         The switch to enable the print of detailed info in the console.
          enable_logger_print      The switch to enable the logger print of detailed info in the console.
          pass_in_logger           The logger object be passed in. (Useful for projects like Swin-Transformer)
          relaxed_tc_check         Integer controling NVIDIA's TC compatibility check level.
                                   0 -> Strict check as spMMA in Ampere (p.size()[0] % 8 == 0, p.size()[1] % 16 == 0).
                                   1 -> The second dim of weight should meed mod 16 requirement (p.size()[1] % 16 == 0).
                                   2 -> No check for weight shape.

        """
        assert (cls.__model is None), "ASP has been initialized already."
        cls.__model = model
        cls.__verbosity = verbosity
        cls.__enable_raw_print = enable_raw_print
        cls.__enable_logger_print = enable_logger_print
        cls.__pass_in_logger = pass_in_logger
        cls.__relaxed_tc_check = relaxed_tc_check

        if isinstance(mask_calculator, str):
            def create_mask_from_pattern(param):
                return create_mask(param, mask_calculator).bool()
            cls.__calculate_mask = create_mask_from_pattern
        else:
            cls.__calculate_mask = mask_calculator #user defined function

        # function to extract variables that will be sparsified. 
        # idea is that you will add one of these functions for each module type that can be sparsified.
        if torchvision_imported and torchvision_ops_supported:
            if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) and cls.__enable_raw_print:
                print("[ASP] torchvision is imported, can work smoothly with the MaskRCNN/KeypointRCNN from torchvision.")
            if cls.__enable_logger_print and cls.__pass_in_logger is not None:
                #logging.info("[ASP] torchvision is imported, can work smoothly with the MaskRCNN/KeypointRCNN from torchvision.")
                cls.__pass_in_logger.info("[ASP] torchvision is imported, can work smoothly with the MaskRCNN/KeypointRCNN from torchvision.")
            torchvision_version = str(torchvision.__version__)
            torchvision_version_major = int(torchvision_version.split('.')[0])
            torchvision_version_minor = int(torchvision_version.split('.')[1])
            try:
                torchvision_version_minimum = int(torchvision_version.split('.')[2])
            except ValueError:    # Chong: support the none standard version
                torchvision_version_minimum = torchvision_version.split('.')[2]
            if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) and cls.__enable_raw_print:
                print("[Chong] The torchvision version is: {}, version major is: {}, version minor is: {}, version minimum is: {}".format(torchvision_version, torchvision_version_major, torchvision_version_minor, torchvision_version_minimum))
            if torchvision_version_major == 0 and torchvision_version_minor < 12:
                sparse_parameter_list = {torch.nn.Linear: ['weight'], torch.nn.Conv1d: ['weight'], torch.nn.Conv2d: ['weight'], torch.nn.Conv3d: ['weight'], torchvision.ops.misc.Conv2d: ['weight']}
            else:    # Chong: Torchvision remove APIs that were deprecated before 0.8 (#5386) in 0.12.0, torchvision.ops.misc.Conv2d is removed
                sparse_parameter_list = {torch.nn.Linear: ['weight'], torch.nn.Conv1d: ['weight'], torch.nn.Conv2d: ['weight'], torch.nn.Conv3d: ['weight']}
        else:
            sparse_parameter_list = {torch.nn.Linear: ['weight'], torch.nn.Conv1d: ['weight'], torch.nn.Conv2d: ['weight'], torch.nn.Conv3d: ['weight']}
        if custom_layer_dict: # Update default list to include user supplied custom (layer type : parameter tensor), make sure this tensor type is something ASP knows how to prune
            sparse_parameter_list.update(custom_layer_dict)
            whitelist += list(custom_layer_dict.keys())

        for module_type in whitelist:
            assert (module_type in sparse_parameter_list), "Module %s :: Don't know how to sparsify module." % module_type

        # find all sparse modules, extract sparse parameters and decorate
        def add_sparse_attributes(module_name, module):
            #if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) and cls.__enable_raw_print:
            #    print("[ASP][Chong] module_name: {}, module: {}, module type: {}".format(module_name, module, type(module)))    #Debug for MaskRCNN 'torchvision.ops.misc.Conv2d'
            sparse_parameters = sparse_parameter_list[type(module)]
            for p_name, p in module.named_parameters():
                # if p_name in sparse_parameters and p.requires_grad:
                if p_name in sparse_parameters:
                    if cls.__relaxed_tc_check == 0:
                        # check for NVIDIA's TC compatibility: we check along the horizontal direction
                        if p.dtype == torch.float32 and ((p.size()[0] % 8) != 0 or (p.size()[1] % 16) != 0): #User defines FP32 and APEX internally uses FP16 math
                            if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) and cls.__enable_raw_print:
                                print("[ASP] Auto skipping pruning %s::%s of size=%s and type=%s for sparsity" % (module_name, p_name, str(p.size()), str(p.dtype)))
                            if cls.__enable_logger_print and cls.__pass_in_logger is not None:
                                #logging.info("[ASP] Auto skipping pruning %s::%s of size=%s and type=%s for sparsity" % (module_name, p_name, str(p.size()), str(p.dtype)))
                                cls.__pass_in_logger.info("[ASP] Auto skipping pruning %s::%s of size=%s and type=%s for sparsity" % (module_name, p_name, str(p.size()), str(p.dtype)))
                            continue
                        if p.dtype == torch.float16 and ((p.size()[0] % 8) != 0 or (p.size()[1] % 16) != 0): #For Conv2d dim= K x CRS; we prune along C
                            if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) and cls.__enable_raw_print:
                                print("[ASP] Auto skipping pruning %s::%s of size=%s and type=%s for sparsity" % (module_name, p_name, str(p.size()), str(p.dtype)))
                            if cls.__enable_logger_print and cls.__pass_in_logger is not None:
                                #logging.info("[ASP] Auto skipping pruning %s::%s of size=%s and type=%s for sparsity" % (module_name, p_name, str(p.size()), str(p.dtype)))
                                cls.__pass_in_logger.info("[ASP] Auto skipping pruning %s::%s of size=%s and type=%s for sparsity" % (module_name, p_name, str(p.size()), str(p.dtype)))
                            continue
                    elif cls.__relaxed_tc_check == 1:
                        # Chong: for some pioneering study, we need to relax the NVIDIA's TC compatibility check, for example: GCN
                        if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) and cls.__enable_raw_print:
                            print("[ASP] Relax the NVIDIA's TC compatibility check for pioneering study! Relaxed level: {:}".format(cls.__relaxed_tc_check))
                        if cls.__enable_logger_print and cls.__pass_in_logger is not None:
                            #logging.info("[ASP] Relax the NVIDIA's TC compatibility check for pioneering study! Relaxed level: {:}".format(cls.__relaxed_tc_check))
                            cls.__pass_in_logger.info("[ASP] Relax the NVIDIA's TC compatibility check for pioneering study! Relaxed level: {:}".format(cls.__relaxed_tc_check))
                        if p.dtype == torch.float32 and (p.size()[1] % 16) != 0: #User defines FP32 and APEX internally uses FP16 math
                            if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) and cls.__enable_raw_print:
                                print("[ASP] Auto skipping pruning %s::%s of size=%s and type=%s for sparsity" % (module_name, p_name, str(p.size()), str(p.dtype)))
                            if cls.__enable_logger_print and cls.__pass_in_logger is not None:
                                #logging.info("[ASP] Auto skipping pruning %s::%s of size=%s and type=%s for sparsity" % (module_name, p_name, str(p.size()), str(p.dtype)))
                                cls.__pass_in_logger.info("[ASP] Auto skipping pruning %s::%s of size=%s and type=%s for sparsity" % (module_name, p_name, str(p.size()), str(p.dtype)))
                            continue
                        if p.dtype == torch.float16 and (p.size()[1] % 16) != 0: #For Conv2d dim= K x CRS; we prune along C
                            if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) and cls.__enable_raw_print:
                                print("[ASP] Auto skipping pruning %s::%s of size=%s and type=%s for sparsity" % (module_name, p_name, str(p.size()), str(p.dtype)))
                            if cls.__enable_logger_print and cls.__pass_in_logger is not None:
                                #logging.info("[ASP] Auto skipping pruning %s::%s of size=%s and type=%s for sparsity" % (module_name, p_name, str(p.size()), str(p.dtype)))
                                cls.__pass_in_logger.info("[ASP] Auto skipping pruning %s::%s of size=%s and type=%s for sparsity" % (module_name, p_name, str(p.size()), str(p.dtype)))
                            continue
                    elif cls.__relaxed_tc_check == 2:
                        # Chong: for some pioneering study, we need to relax the NVIDIA's TC compatibility check, for example: GAT
                        if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) and cls.__enable_raw_print:
                            print("[ASP] Relax the NVIDIA's TC compatibility check for pioneering study! Relaxed level: {:}".format(cls.__relaxed_tc_check))
                        if cls.__enable_logger_print and cls.__pass_in_logger is not None:
                            #logging.info("[ASP] Relax the NVIDIA's TC compatibility check for pioneering study! Relaxed level: {:}".format(cls.__relaxed_tc_check))
                            cls.__pass_in_logger.info("[ASP] Relax the NVIDIA's TC compatibility check for pioneering study! Relaxed level: {:}".format(cls.__relaxed_tc_check))

                    if cls.__verbosity >= 3:
                        if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) and cls.__enable_raw_print:
                            print("[ASP] Sparsifying %s::%s of size=%s and type=%s for sparsity" % (module_name, p_name, str(p.size()), str(p.dtype)))
                        if cls.__enable_logger_print and cls.__pass_in_logger is not None:
                            #logging.info("[ASP] Sparsifying %s::%s of size=%s and type=%s for sparsity" % (module_name, p_name, str(p.size()), str(p.dtype)))
                            cls.__pass_in_logger.info("[ASP] Sparsifying %s::%s of size=%s and type=%s for sparsity" % (module_name, p_name, str(p.size()), str(p.dtype)))
                    
                    mask = torch.ones_like(p).bool()
                    buffname = p_name.split(".")[-1] # buffer names cannot contain "."
                    module.register_buffer('__%s_mma_mask' % buffname, mask)
                    if allow_recompute_mask:
                        pruned = torch.zeros_like(p).cuda()
                        module.register_buffer('__%s_mma_pruned_p' % buffname, pruned)
                    else:
                        pruned = None
                    cls.__sparse_parameters.append((module_name, module, p_name, p, mask, pruned))
                else:
                    if cls.__verbosity >= 3:
                        if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) and cls.__enable_raw_print:
                            print("[ASP] Not sparsifying %s::%s of size=%s and type=%s" % (module_name, p_name, str(p.size()), str(p.dtype)))
                        if cls.__enable_logger_print and cls.__pass_in_logger is not None:
                            #logging.info("[ASP] Not sparsifying %s::%s of size=%s and type=%s" % (module_name, p_name, str(p.size()), str(p.dtype)))
                            cls.__pass_in_logger.info("[ASP] Not sparsifying %s::%s of size=%s and type=%s" % (module_name, p_name, str(p.size()), str(p.dtype)))

        for name, sparse_module in eligible_modules(model, tuple(whitelist), allowed_layer_names, disallowed_layer_names):
            add_sparse_attributes(name, sparse_module)

    @classmethod
    def already_init_asp_model(cls):
        """Call this method to check whether ASP has been initialized already.
        """
        if cls.__model is None:
            if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) and cls.__verbosity >= 3:
                print("[ASP] ASP has not been initialized.")
            return False
        else:
            if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) and cls.__verbosity >= 3:
                print("[ASP] ASP has been initialized already.")
            return True

    @classmethod
    def reset_init_asp_status(cls):
        """Call this method to reset the ASP status, from has been initialized already to not initialized.
        """
        cls.__model = None
        if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) and cls.__verbosity >= 3:
            print("[ASP] ASP status has been reset as not initialized.")

    @classmethod
    def init_optimizer_for_pruning(cls, optimizer):
        """Call this method to monkey patch optimizer step function so that masks can be applied to
        gradients and weights during training.
        You must call init_model_for_pruning(...) before calling init_optimizer_for_pruning(...)
        """
        assert (cls.__optimizer is None), "ASP has initialized optimizer already."
        assert (cls.__calculate_mask is not None), "Called ASP.init_optimizer_for_pruning before ASP.init_model_for_pruning."

        # store pointer to original optimizer step method
        cls.__optimizer = optimizer
        cls.__optimizer.__step = optimizer.step

        def __step(opt_self, *args, **kwargs):
            # prune gradients before step method
            with torch.no_grad():
                for module_name, module, p_name, p, mask, pruned in cls.__sparse_parameters:
                    if p.grad is not None: #thx pjudd
                    # Fixing mask multiplication with grad tensors
                    # Grads can be None type. Adding this fix to skip multiplication with masks if this is the case.
                        try:
                            p.grad.mul_(mask)
                        except RuntimeError:    # RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
                            mask = mask.cuda()
                            p.grad.mul_(mask)
                            # Chong: comment out as this cannot release GPU memory cache.
                            #mask = mask.cpu()    # Move the mask back to CPU, otherwise CUDA will out of memory. However, may make the training slower.

            # call original optimizer step method
            rval = opt_self.__step(*args, **kwargs)
            # prune parameters after step method
            with torch.no_grad():
                for module_name, module, p_name, p, mask, pruned in cls.__sparse_parameters:
                    try:
                        p.mul_(mask)
                    except RuntimeError:    # RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
                        mask = mask.cuda()
                        p.mul_(mask)
                        # Chong: comment out as this cannot release GPU memory cache.
                        #mask = mask.cpu()    # Move the mask back to CPU, otherwise CUDA will out of memory. However, may make the training slower.
            return rval
        cls.__optimizer.step = types.MethodType(__step, cls.__optimizer)

    @classmethod
    def enable_sparsity(cls):
        """This function is deprecated. Please switch to compute_sparse_masks, which has identical behavior."""
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print("[ASP] `enable_sparsity` function is deprecated and has been renamed `compute_sparse_masks`.")
        return cls.compute_sparse_masks()

    @classmethod
    def compute_sparse_masks(cls, manually_off_print=False, mask_diff_statistics=False):
        """Call this method to enable sparsity.
        If init(...) was called with allow_recompute_mask=False AND sparsity is disabled, pruned field can be None.
        """
        with torch.no_grad():
            for module_name, module, p_name, p, mask, pruned in cls.__sparse_parameters:
                if mask.sum() < mask.numel(): # when recalculating masks
                    # restore dense parameter if allow_recompute_mask is enabled
                    assert (pruned is not None), "Unable to restore dense parameter because allow_recompute_mask == False"
                    p.add_(pruned.cuda())

                if mask_diff_statistics:    # Chong: make the statistics for dynamic mask diff
                    original_mask = mask.clone()

                try:
                    mask.set_(cls.__calculate_mask(p))

                    if pruned is not None: # pruned weights are on cpu
                        pruned.set_((p * (~mask)).cuda())

                    p.mul_(mask) # in-place multiplication, so pruned weights are 0-values
                except RuntimeError:    # RuntimeError: Attempted to set the storage of a tensor on device "cpu" to a storage on different device "cuda:0".  This is no longer allowed; the devices must match.
                    mask_gpu = mask.cuda()
                    mask_gpu.set_(cls.__calculate_mask(p))

                    if pruned is not None: # pruned weights are on cpu
                        pruned.set_((p * (~mask_gpu)).cuda())

                    p.mul_(mask_gpu) # in-place multiplication, so pruned weights are 0-values
                    mask.set_(mask_gpu.cuda())

                if mask_diff_statistics:    # Chong: make the statistics for dynamic mask diff
                    mask_diff = mask.eq(original_mask)    # Chong: if the position that original_mask equals to mask, then return 1; otherwise return 0.
                    if cls.__verbosity >= 2:
                        if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) and cls.__enable_raw_print:
                            print("[ASP] There are %.4f%% difference in learnable sparse mask for %s::%s" % (100.0-100.0*mask_diff.sum()/mask_diff.numel(), module_name, p_name))
                        if cls.__enable_logger_print and cls.__pass_in_logger is not None:
                            cls.__pass_in_logger.info("[ASP] There are %.4f%% difference in learnable sparse mask for %s::%s" % (100.0-100.0*mask_diff.sum()/mask_diff.numel(), module_name, p_name))

                if cls.__verbosity >= 2:
                    if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) and cls.__enable_raw_print:
                        print("[ASP] Enabled %.2f%% sparsity for %s::%s of size=%s and type=%s with magnitude %s" % (100.0-100.0*mask.sum()/mask.numel(), module_name, p_name, str(p.size()), str(p.dtype), torch.sum(torch.abs(p))))
                    if cls.__enable_logger_print and cls.__pass_in_logger is not None:
                        #logging.info("[ASP] Enabled %.2f%% sparsity for %s::%s of size=%s and type=%s with magnitude %s" % (100.0-100.0*mask.sum()/mask.numel(), module_name, p_name, str(p.size()), str(p.dtype), torch.sum(torch.abs(p))))
                        cls.__pass_in_logger.info("[ASP] Enabled %.2f%% sparsity for %s::%s of size=%s and type=%s with magnitude %s" % (100.0-100.0*mask.sum()/mask.numel(), module_name, p_name, str(p.size()), str(p.dtype), torch.sum(torch.abs(p))))

    @classmethod
    def disable_sparsity(cls):
        """Call this method to disable sparsity."""
        """This function is deprecated. Please switch to restore_pruned_weights, which has identical behavior."""
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print("[ASP] `disable_sparsity` function is deprecated and has been renamed `restore_pruned_weights`.")
        return cls.restore_pruned_weights()

    @classmethod
    def restore_pruned_weights(cls):
        """Call this method to disable sparsity and restore all weights.
        This will only work if init(...) was called with allow_recompute=True.
        """
        with torch.no_grad():
            for module_name, module, p_name, p, mask, pruned in cls.__sparse_parameters:
                if mask.sum() < mask.numel():
                    assert (pruned is not None), "Unable to restore dense parameter because allow_recompute_mask == False"
                    p.add_(pruned.cuda())
                    mask.fill_(1)
                    pruned.zero_()
                    if cls.__verbosity >= 2:
                        if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) and cls.__enable_raw_print:
                            print("[ASP] Disabled sparsity for %s::%s (dense weights restored)" % (module_name, p_name))
                        if cls.__enable_logger_print and cls.__pass_in_logger is not None:
                            #logging.info("[ASP] Disabled sparsity for %s::%s (dense weights restored)" % (module_name, p_name))
                            cls.__pass_in_logger.info("[ASP] Disabled sparsity for %s::%s (dense weights restored)" % (module_name, p_name))

    @classmethod
    def sparsity_is_enabled(cls):
        """This function is deprecated. Please switch to is_sparsity_enabled, which has identical behavior."""
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print("[ASP] `sparsity_is_enabled` function is deprecated and has been renamed `is_sparsity_enabled`.")
        return cls.is_sparsity_enabled()

    @classmethod
    def is_sparsity_enabled(cls):
        """Call this method to determine if sparsity is enabled in the model.
        The typical use case is right after checkpoint has been loaded.
        """
        total,sp100,sp50 = 0,0,0
        if cls.__verbosity >= 3:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print("[ASP] Checking whether the sparsity is enabled or disabled in the model")
            if cls.__enable_logger_print and cls.__pass_in_logger is not None:
                #logging.info("[ASP] Checking whether the sparsity is enabled or disabled in the model")
                cls.__pass_in_logger.info("[ASP] Checking whether the sparsity is enabled or disabled in the model")
        for module_name, module, p_name, p, mask, pruned in cls.__sparse_parameters:
            total += 1
            mask_sum = mask.sum()
            mask_numel = mask.numel()
            if mask_sum == mask_numel:
                sp100 += 1
            elif mask_sum*2 == mask_numel:
                sp50 += 1
            if cls.__verbosity >= 3:
                if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) and cls.__enable_raw_print:
                    print("[ASP] Confirmed %.2f%% sparsity for %s::%s of size=%s and type=%s, mask_sum=%s, mask_numel=%s" % (100.0-100.0*mask.sum()/mask.numel(), module_name, p_name, str(p.size()), str(p.dtype), mask_sum, mask_numel))
                if cls.__enable_logger_print and cls.__pass_in_logger is not None:
                    #logging.info("[ASP] Confirmed %.2f%% sparsity for %s::%s of size=%s and type=%s, mask_sum=%s, mask_numel=%s" % (100.0-100.0*mask.sum()/mask.numel(), module_name, p_name, str(p.size()), str(p.dtype), mask_sum, mask_numel))
                    cls.__pass_in_logger.info("[ASP] Confirmed %.2f%% sparsity for %s::%s of size=%s and type=%s, mask_sum=%s, mask_numel=%s" % (100.0-100.0*mask.sum()/mask.numel(), module_name, p_name, str(p.size()), str(p.dtype), mask_sum, mask_numel))

        assert (total == sp100 or total == sp50), "Inconsistent model sparsity"
        if total == sp100:
            return False
        elif total == sp50:
            return True

    @classmethod
    def prune_trained_model(cls, model, optimizer):
        ## add mask buffers to model (init_model_for_pruning), augment optimizer (init_optimizer_for_pruning) and compute masks (enable_sparsity)
        # add mask buffers to model (init_model_for_pruning), augment optimizer (init_optimizer_for_pruning) and compute masks (compute_sparse_masks)
        cls.init_model_for_pruning(model, mask_calculator="m4n2_1d", verbosity=2, whitelist=[torch.nn.Linear, torch.nn.Conv2d], allow_recompute_mask=False)
        cls.init_optimizer_for_pruning(optimizer)
        #cls.enable_sparsity()     #enable_sparsity function is deprecated
        cls.compute_sparse_masks()

    @classmethod
    def sparsity_statistics(cls, disable_mask_statistics=False):
        """Call this method to make the statistics of the sparsity info of the model.
        """
        all_zero_element_num_weight = 0.0
        with torch.no_grad():
            for module_name, module, p_name, p, mask, pruned in cls.__sparse_parameters:
                if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) and cls.__enable_raw_print:
                    print("[ASP][sparsity_statistics] module_name: {:}, module: {:}, module type: {:}".format(module_name, module, type(module)))
                if cls.__enable_logger_print and cls.__pass_in_logger is not None:
                    #logging.info("[ASP][sparsity_statistics] module_name: {:}, module: {:}, module type: {:}".format(module_name, module, type(module)))
                    cls.__pass_in_logger.info("[ASP][sparsity_statistics] module_name: {:}, module: {:}, module type: {:}".format(module_name, module, type(module)))
                #Summarize the sparsity info of weight
                non_zero_fraction_weight = float(p.nonzero().size(0)) / p.numel()
                zero_element_num_weight = p.numel() - p.nonzero().size(0)
                all_zero_element_num_weight += zero_element_num_weight
                if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) and cls.__enable_raw_print:
                    print("[ASP][sparsity_statistics][weight] \tNone zero fraction: {:.4f}, Total element num: {:}, Zero element num: {:}".format(non_zero_fraction_weight, p.numel(), zero_element_num_weight))
                if cls.__enable_logger_print and cls.__pass_in_logger is not None:
                    #logging.info("[ASP][sparsity_statistics][weight] \tNone zero fraction: {:.4f}, Total element num: {:}, Zero element num: {:}".format(non_zero_fraction_weight, p.numel(), zero_element_num_weight))
                    cls.__pass_in_logger.info("[ASP][sparsity_statistics][weight] \tNone zero fraction: {:.4f}, Total element num: {:}, Zero element num: {:}".format(non_zero_fraction_weight, p.numel(), zero_element_num_weight))
                abs_element_sum_weight = (torch.abs(p)).sum(dtype=torch.float64)
                if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) and cls.__enable_raw_print:
                    print("[ASP][sparsity_statistics][weight] \tElement abs sum: {:} ({:})".format(abs_element_sum_weight, module_name))
                if cls.__enable_logger_print and cls.__pass_in_logger is not None:
                    #logging.info("[ASP][sparsity_statistics][weight] \tElement abs sum: {:} ({:})".format(abs_element_sum_weight, module_name))
                    cls.__pass_in_logger.info("[ASP][sparsity_statistics][weight] \tElement abs sum: {:} ({:})".format(abs_element_sum_weight, module_name))

                if not disable_mask_statistics:    # Chong: to avoid the RuntimeError: "abs_cpu" not implemented for 'Bool'
                    #Summarize the sparsity info of mask
                    non_zero_fraction_mask = float(mask.nonzero().size(0)) / mask.numel()
                    zero_element_num_mask = mask.numel() - mask.nonzero().size(0)
                    if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) and cls.__enable_raw_print:
                        print("[ASP][sparsity_statistics][mask] \tNone Zero fraction: {:.4f}, Total element num: {:}, Zero element num: {:}".format(non_zero_fraction_mask, mask.numel(), zero_element_num_mask))
                    if cls.__enable_logger_print and cls.__pass_in_logger is not None:
                        #logging.info("[ASP][sparsity_statistics][mask] \tNone Zero fraction: {:.4f}, Total element num: {:}, Zero element num: {:}".format(non_zero_fraction_mask, mask.numel(), zero_element_num_mask))
                        cls.__pass_in_logger.info("[ASP][sparsity_statistics][mask] \tNone Zero fraction: {:.4f}, Total element num: {:}, Zero element num: {:}".format(non_zero_fraction_mask, mask.numel(), zero_element_num_mask))
                    #abs_element_sum_mask = (torch.abs(mask)).sum()
                    abs_element_sum_mask = mask.sum()    # Chong: if the mask element is bool, seems no need to go through the abs function, then the RuntimeError can be avoid.
                    if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) and cls.__enable_raw_print:
                        print("[ASP][sparsity_statistics][mask] \tElement abs sum: {:}".format(abs_element_sum_mask))
                    if cls.__enable_logger_print and cls.__pass_in_logger is not None:
                        #logging.info("[ASP][sparsity_statistics][mask] \tElement abs sum: {:}".format(abs_element_sum_mask))
                        cls.__pass_in_logger.info("[ASP][sparsity_statistics][mask] \tElement abs sum: {:}".format(abs_element_sum_mask))
        all_zero_element_num_weight /= 1e6
        print("all_zero_element_num_weight:", all_zero_element_num_weight)
