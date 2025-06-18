from .surrogate_factory import register_surrogate
import torch


@register_surrogate('adaptive_aggregation_f1_8')
class AdaptiveAggregationF1Surrogate8:
    def __init__(self, **kwargs) -> None:
        self.name = kwargs.get('name', 'surrogate')
        self.weight = kwargs.get('weight', 1.0)
        self.average = kwargs.get('average', None)
        self.upper_bound = kwargs.get('upper_bound', 1.0)
        self.use_max = kwargs.get('use_max', False)
        self.lambda_global = kwargs.get("lambda_global", 1.0)
        self.lambda_peer = kwargs.get("lambda_peer", 1.0)
        self.temperature = kwargs.get("temperature", 2.0)
        self.weights_list = []

    def set_weights(self, weights):
        self.weights_list = weights

    def _wasserstein_distance_global(self, p, q):
        assert p.shape[1] == q.shape[1], 'Distributions must have same number of classes'
        F_p = torch.cumsum(torch.mean(p, dim=0), dim=0)
        F_q = torch.cumsum(torch.mean(q, dim=0), dim=0)
        return torch.sum(torch.abs(F_p - F_q)).to(p.device)

    def __call__(self, **kwargs):
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        probabilities = kwargs.get('probabilities')
        teacher_logits_list = kwargs.get('teacher_logits_list')
        output_distribution = kwargs.get('output_distribution')
        teacher_probabilities_list = kwargs.get('teacher_probabilities')

        assert logits is not None and labels is not None, 'logits and labels must be provided'
        assert probabilities is not None, 'probabilities must be provided'
        assert teacher_logits_list is not None, 'teacher_logits_list must be provided'
        assert output_distribution is not None, 'output_distribution must be provided'
        assert len(self.weights_list) == len(teacher_logits_list), \
            'weights_list must match length of teacher_logits_list'

        if torch.isnan(probabilities).any():
            raise ValueError('Probabilities contain NaN')
        for t in teacher_logits_list:
            if torch.isnan(t).any():
                raise ValueError("Teacher logits contain NaN")
            # print('Teacher logits shape:', t.shape)
            # print('Student logits shape:', logits.shape)
            assert t.shape == logits.shape, 'Teacher and student logits must match in shape'

        if isinstance(self.weights_list, torch.Tensor):
            raw_weights = self.weights_list
        else:
            assert isinstance(self.weights_list, (list, tuple)
                              ), "weights_list must be a list or tuple"
            raw_weights = torch.tensor(
                self.weights_list, device=teacher_logits_list[0].device)
            raw_weights = raw_weights.to(dtype=teacher_logits_list[0].dtype)

        # raw_weights = torch.tensor(self.weights_list, dtype=torch.float32, device=teacher_logits_list[0].device)
        weights_tensor = torch.softmax(raw_weights / 0.5, dim=0)
        # print('Weights tensor:', weights_tensor)
        # print('Weights:', weights_tensor)
        if isinstance(teacher_logits_list, torch.Tensor):
            stacked_logits = teacher_logits_list  # già [T, B, C]
        else:
            assert isinstance(teacher_logits_list, (list, tuple)
                              ), "teacher_logits_list must be a list or tuple"
            stacked_logits = torch.stack(teacher_logits_list, dim=0)

        weighted_logits = torch.einsum(
            'tbc,t->bc', stacked_logits, weights_tensor)  # [B, C]

        # Teacher consensus: softmax over averaged logits
        teacher_ensemble_distribution = torch.softmax(
            weighted_logits / self.temperature, dim=1)  # [B, C]
        # student_distribution = torch.softmax(logits, dim=1)  # [B, C]
        # Student prediction: log-softmax
        student_log_probs = torch.log_softmax(
            logits / self.temperature, dim=1)  # [B, C]

        loss = torch.nn.functional.kl_div(student_log_probs, teacher_ensemble_distribution,
                                          reduction='batchmean', log_target=False) * (self.temperature ** 2)
        # loss = self._wasserstein_distance_global(student_distribution, teacher_ensemble_distribution)
        # print('Loss:', loss.item())
        return loss


@register_surrogate('adaptive_aggregation_f1_8_max')
class AdaptiveAggregationF1Surrogate8Max:
    def __init__(self, **kwargs) -> None:
        self.name = kwargs.get('name', 'surrogate')
        self.weight = kwargs.get('weight', 1.0)
        self.average = kwargs.get('average', None)
        self.upper_bound = kwargs.get('upper_bound', 1.0)
        self.use_max = kwargs.get('use_max', False)
        self.lambda_global = kwargs.get("lambda_global", 1.0)
        self.lambda_peer = kwargs.get("lambda_peer", 1.0)
        self.temperature = kwargs.get("temperature", 2.0)
        self.weights_list = []

    def set_weights(self, weights):
        self.weights_list = weights

    def _wasserstein_distance_global(self, p, q):
        assert p.shape[1] == q.shape[1], 'Distributions must have same number of classes'
        F_p = torch.cumsum(torch.mean(p, dim=0), dim=0)
        F_q = torch.cumsum(torch.mean(q, dim=0), dim=0)
        return torch.sum(torch.abs(F_p - F_q)).to(p.device)

    def __call__(self, **kwargs):
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        probabilities = kwargs.get('probabilities')
        teacher_logits_list = kwargs.get('teacher_logits_list')
        output_distribution = kwargs.get('output_distribution')
        teacher_probabilities_list = kwargs.get('teacher_probabilities')

        assert logits is not None and labels is not None, 'logits and labels must be provided'
        assert probabilities is not None, 'probabilities must be provided'
        assert teacher_logits_list is not None, 'teacher_logits_list must be provided'
        assert output_distribution is not None, 'output_distribution must be provided'
        assert len(self.weights_list) == len(teacher_logits_list), \
            'weights_list must match length of teacher_logits_list'

        if torch.isnan(probabilities).any():
            raise ValueError('Probabilities contain NaN')
        for t in teacher_logits_list:
            if torch.isnan(t).any():
                raise ValueError("Teacher logits contain NaN")
            # print('Teacher logits shape:', t.shape)
            # print('Student logits shape:', logits.shape)
            assert t.shape == logits.shape, 'Teacher and student logits must match in shape'

        if isinstance(self.weights_list, torch.Tensor):
            raw_weights = self.weights_list
        else:
            assert isinstance(self.weights_list, (list, tuple)
                              ), "weights_list must be a list or tuple"
            raw_weights = torch.tensor(
                self.weights_list, device=teacher_logits_list[0].device)
            raw_weights = raw_weights.to(dtype=teacher_logits_list[0].dtype)

        # raw_weights = torch.tensor(self.weights_list, dtype=torch.float32, device=teacher_logits_list[0].device)
        weights_tensor = torch.softmax(raw_weights / self.temperature, dim=0)
        # print('Weights tensor:', weights_tensor)
        # print('Weights:', weights_tensor)
        if isinstance(teacher_logits_list, torch.Tensor):
            stacked_logits = teacher_logits_list  # già [T, B, C]
        else:
            assert isinstance(teacher_logits_list, (list, tuple)
                              ), "teacher_logits_list must be a list or tuple"
            stacked_logits = torch.stack(teacher_logits_list, dim=0)

        weighted_logits = torch.einsum(
            'tbc,t->bc', stacked_logits, weights_tensor)  # [B, C]

        # Teacher consensus: softmax over averaged logits
        teacher_ensemble_distribution = torch.softmax(
            weighted_logits / self.temperature, dim=1)  # [B, C]
        # student_distribution = torch.softmax(logits, dim=1)  # [B, C]
        # Student prediction: log-softmax
        student_log_probs = torch.log_softmax(
            logits / self.temperature, dim=1)  # [B, C]

        loss = torch.nn.functional.kl_div(student_log_probs, teacher_ensemble_distribution,
                                          reduction='batchmean', log_target=False) * (self.temperature ** 2)
        # loss = self._wasserstein_distance_global(student_distribution, teacher_ensemble_distribution)
        # print('Loss:', loss.item())
        return -loss


@register_surrogate('adaptive_aggregation_f1_10')
class AdaptiveAggregationF1Surrogate10:
    def __init__(self, **kwargs) -> None:
        self.name = kwargs.get('name', 'surrogate')
        self.weight = kwargs.get('weight', 1.0)
        self.average = kwargs.get('average', None)
        self.upper_bound = kwargs.get('upper_bound', 1.0)
        self.use_max = kwargs.get('use_max', False)
        self.lambda_global = kwargs.get("lambda_global", 1.0)
        self.temperature = kwargs.get("temperature", 2.0)
        self.weights_list = []

    def set_weights(self, weights):
        self.weights_list = weights

    def _distillation_loss(self, logits, teacher_logits_list,
                           probabilities, teacher_probabilities_list, labels):
        labels = labels.float()
        student_prob = probabilities[:, 1]

        # ==== 1. Crea ensemble logits (media)
        if isinstance(teacher_logits_list, torch.Tensor):
            stacked = teacher_logits_list  # [T, B, C]
        else:
            stacked = torch.stack(teacher_logits_list, dim=0)  # [T, B, C]
        ensemble_logits = stacked.mean(dim=0)  # [B, C]
        # ensemble_logits = torch.einsum('tbc,t->bc', stacked, weights_tensor)  # [B, C]

        # ==== 2. Costruisci FP/FN mask
        with torch.no_grad():
            teacher_probs = torch.nn.functional.softmax(
                ensemble_logits, dim=1)  # [B, C]
            teacher_pred = torch.argmax(teacher_probs, dim=1).float()

        fn_mask = labels * teacher_pred * (1 - student_prob)
        fp_mask = (1 - labels) * (1 - teacher_pred) * student_prob
        correction_mask = (fn_mask + fp_mask).unsqueeze(1)  # [B, 1]

        # ==== 3. Teacher soft + confidenza
        T = self.temperature

        student_log_soft = torch.log_softmax(logits / T, dim=1)
        teacher_soft = torch.softmax(ensemble_logits / T, dim=1).detach()
        # teacher_conf = teacher_soft.max(dim=1)[0].unsqueeze(1)  # [B, 1]

        # ==== 4. KL distillazione focalizzata + pesata per confidenza
        kl = torch.nn.functional.kl_div(
            student_log_soft, teacher_soft, reduction='none').sum(dim=1, keepdim=True)
        kl = kl * (T * T)

        mask = correction_mask  # * teacher_conf  # [B, 1], pesata
        loss = (kl * mask).sum() / (mask.sum() + 1e-8)

        return loss

    def __call__(self, **kwargs):
        logits = kwargs.get('logits')                           # [B, C]
        labels = kwargs.get('labels')                           # [B]
        probabilities = kwargs.get('probabilities')             # [B, C]
        teacher_logits_list = kwargs.get(
            'teacher_logits_list')  # list of [B, C]
        teacher_probabilities_list = kwargs.get(
            'teacher_probabilities')  # list of [B, C]

        assert logits is not None and labels is not None
        assert probabilities is not None
        assert teacher_logits_list is not None
        assert teacher_probabilities_list is not None

        if torch.isnan(probabilities).any():
            raise ValueError('Probabilities contiene NaN!')

        # ==== CE pesata per la parte supervisionata
        B, C = probabilities.shape
        labels_long = labels.long()
        class_counts = torch.bincount(labels_long, minlength=C).float() + 1e-6
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * C
        ce_loss = torch.nn.functional.cross_entropy(
            logits, labels_long, weight=class_weights.to(logits.device))

        # ==== Distillazione
        if len(teacher_logits_list) == 0:
            return ce_loss

        distill_loss = self._distillation_loss(logits, teacher_logits_list,
                                               probabilities, teacher_probabilities_list, labels)

        return ce_loss + self.lambda_global * distill_loss


@register_surrogate('batch_adaptive_aggregation_f1_10')
class BatchAdaptiveAggregationF1Surrogate10:
    def __init__(self, **kwargs) -> None:
        self.name = kwargs.get('name', 'surrogate')
        self.weight = kwargs.get('weight', 1.0)
        self.average = kwargs.get('average', None)
        self.upper_bound = kwargs.get('upper_bound', 1.0)
        self.use_max = kwargs.get('use_max', False)
        self.lambda_global = kwargs.get("lambda_global", 1.0)
        self.temperature = kwargs.get("temperature", 2.0)

    def _distillation_loss(self, logits, teacher_logits_list,
                           probabilities, teacher_probabilities_list, labels):
        labels = labels.float()
        student_prob = probabilities[:, 1]

        # Ensemble logits
        if isinstance(teacher_logits_list, torch.Tensor):
            stacked = teacher_logits_list  # [T, B, C]
        else:
            stacked = torch.stack(teacher_logits_list, dim=0)  # [T, B, C]
        ensemble_logits = stacked.mean(dim=0)  # [B, C]

        with torch.no_grad():
            teacher_probs = teacher_probabilities_list.mean(dim=0)  # [B, C]
            teacher_pred = torch.argmax(teacher_probs, dim=1).float()

        fn_mask = labels * teacher_pred * (1 - student_prob)
        fp_mask = (1 - labels) * (1 - teacher_pred) * student_prob
        correction_mask = (fn_mask + fp_mask).unsqueeze(1)  # [B, 1]

        T = self.temperature
        student_log_soft = torch.log_softmax(logits / T, dim=1)
        teacher_soft = torch.softmax(ensemble_logits / T, dim=1).detach()
        # teacher_conf = teacher_soft.max(dim=1)[0].unsqueeze(1)  # [B, 1]

        kl = torch.nn.functional.kl_div(
            student_log_soft, teacher_soft, reduction='none').sum(dim=1, keepdim=True)
        kl = kl * (T * T)

        mask = correction_mask   # [B, 1]
        masked_kl = (kl * mask).squeeze(1)  # [B]

        denom = (mask.squeeze(1) + 1e-8)  # [B]
        distill_loss_per_sample = masked_kl / denom.clamp(min=1e-8)  # [B]

        return distill_loss_per_sample

    def __call__(self, **kwargs):
        logits = kwargs.get('logits')                           # [B, C]
        labels = kwargs.get('labels')                           # [B]
        probabilities = kwargs.get('probabilities')             # [B, C]
        teacher_logits_list = kwargs.get(
            'teacher_logits_list')  # list of [B, C]
        teacher_probabilities_list = kwargs.get(
            'teacher_probabilities')  # list of [B, C]

        assert logits is not None and labels is not None
        assert probabilities is not None
        assert teacher_logits_list is not None
        assert teacher_probabilities_list is not None

        if torch.isnan(probabilities).any():
            raise ValueError('Probabilities contiene NaN!')

        B, C = probabilities.shape
        labels_long = labels.long()

        # Cross entropy per esempio (senza media)
        class_counts = torch.bincount(labels_long, minlength=C).float() + 1e-6
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * C
        ce_loss_per_sample = torch.nn.functional.cross_entropy(
            logits, labels_long, weight=class_weights.to(logits.device), reduction='none'
        )  # [B]

        # ==== Distillazione
        if len(teacher_logits_list) == 0:
            return ce_loss_per_sample  # shape [B]

        distill_loss_per_sample = self._distillation_loss(
            logits, teacher_logits_list, probabilities, teacher_probabilities_list, labels
        )  # [B]

        # [B]
        return ce_loss_per_sample + self.lambda_global * distill_loss_per_sample
