"""SeqNN regression metrics."""

import pdb
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.utils import metrics_utils

################################################################################
# Losses
################################################################################
# def MeanSquaredErrorSpecificity(y_true, y_pred, spec_weight=1):
#   mse_term = tf.keras.losses.mean_squared_error(y_pred, y_true)

#   yn_true = y_true - tf.math.reduce_mean(y_true, axis=-1, keepdims=True)
#   yn_pred = y_pred - tf.math.reduce_mean(y_pred, axis=-1, keepdims=True)
#   spec_term = tf.keras.losses.mean_squared_error(yn_pred, yn_true)

#   return mse_term + spec_weight*spec_term

def mean_squared_error_udot(y_true, y_pred, udot_weight=1):
  mse_term = tf.keras.losses.mean_squared_error(y_true, y_pred)

  yn_true = y_true - tf.math.reduce_mean(y_true, axis=-1, keepdims=True)
  yn_pred = y_pred - tf.math.reduce_mean(y_pred, axis=-1, keepdims=True)
  udot_term = -tf.reduce_mean(yn_true * yn_pred, axis=-1)

  return mse_term + udot_weight*udot_term

class MeanSquaredErrorUDot(LossFunctionWrapper):
  def __init__(self, udot_weight=1, reduction=losses_utils.ReductionV2.AUTO, name='mse_udot'):
    self.udot_weight = udot_weight
    mse_udot = lambda yt, yp: mean_squared_error_udot(yt, yp, self.udot_weight)
    super(MeanSquaredErrorUDot, self).__init__(
        mse_udot, name=name, reduction=reduction)


def poisson_kl(y_true, y_pred, kl_weight=1, epsilon=1e-3):
  # poisson loss
  poisson_term = tf.keras.losses.poisson(y_true, y_pred)

  # add epsilon to protect against all tiny values
  y_true += epsilon
  y_pred += epsilon

  # normalize to sum to one
  yn_true = y_true / tf.math.reduce_sum(y_true, axis=-1, keepdims=True)
  yn_pred = y_pred / tf.math.reduce_sum(y_pred, axis=-1, keepdims=True)

  # kl term
  kl_term = tf.keras.losses.kl_divergence(yn_true, yn_pred)

  # weighted combination
  return poisson_term + kl_weight*kl_term

class PoissonKL(LossFunctionWrapper):
  def __init__(self, kl_weight=1, reduction=losses_utils.ReductionV2.AUTO, name='poisson_kl'):
    self.kl_weight = kl_weight
    pois_kl = lambda yt, yp: poisson_kl(yt, yp, self.kl_weight)
    super(PoissonKL, self).__init__(
        pois_kl, name=name, reduction=reduction)


def poisson_multinomial(y_true, y_pred, total_weight=1, epsilon=1e-6, rescale=False):
  seq_len = y_true.shape[1]

  # add epsilon to protect against tiny values
  y_true += epsilon
  y_pred += epsilon

  # sum across lengths
  s_true = tf.math.reduce_sum(y_true, axis=-2, keepdims=True)
  s_pred = tf.math.reduce_sum(y_pred, axis=-2, keepdims=True)

  # normalize to sum to one
  p_pred = y_pred / s_pred

  # total count poisson loss
  poisson_term = tf.keras.losses.poisson(s_true, s_pred) # B x T
  poisson_term /= seq_len

  # multinomial loss
  pl_pred = tf.math.log(p_pred) # B x L x T
  multinomial_dot = -tf.math.multiply(y_true, pl_pred) # B x L x T
  multinomial_term = tf.math.reduce_sum(multinomial_dot, axis=-2) # B x T
  multinomial_term /= seq_len

  # normalize to scale of 1:1 term ratio
  loss_raw = multinomial_term + total_weight * poisson_term
  if rescale:
    loss_rescale = loss_raw*2/(1 + total_weight)
  else:
    loss_rescale = loss_raw

  return loss_rescale

class PoissonMultinomial(LossFunctionWrapper):
  def __init__(self, total_weight=1, reduction=losses_utils.ReductionV2.AUTO, name='poisson_multinomial'):
    self.total_weight = total_weight
    pois_mn = lambda yt, yp: poisson_multinomial(yt, yp, self.total_weight)
    super(PoissonMultinomial, self).__init__(
        pois_mn, name=name, reduction=reduction)


################################################################################
# Metrics
################################################################################
class SeqAUC(tf.keras.metrics.AUC):
  def __init__(self, curve='ROC', name=None, summarize=True, **kwargs):
    if name is None:
      if curve == 'ROC':
        name = 'auroc'
      elif curve == 'PR':
        name = 'auprc'
    super(SeqAUC, self).__init__(curve=curve, name=name, multi_label=True, **kwargs)
    self._summarize = summarize
    

  def update_state(self, y_true, y_pred, **kwargs):
    """Flatten sequence length before update."""

    # flatten batch and sequence length
    num_targets = y_pred.shape[-1]
    y_true = tf.reshape(y_true, (-1,num_targets))
    y_pred = tf.reshape(y_pred, (-1,num_targets))

    # update
    super(SeqAUC, self).update_state(y_true, y_pred, **kwargs)


  def interpolate_pr_auc(self):
    """Add option to remove summary."""
    dtp = self.true_positives[:self.num_thresholds -
                              1] - self.true_positives[1:]
    p = tf.math.add(self.true_positives, self.false_positives)
    dp = p[:self.num_thresholds - 1] - p[1:]
    prec_slope = tf.math.divide_no_nan(
        dtp, tf.maximum(dp, 0), name='prec_slope')
    intercept = self.true_positives[1:] - tf.multiply(prec_slope, p[1:])

    safe_p_ratio = tf.where(
        tf.logical_and(p[:self.num_thresholds - 1] > 0, p[1:] > 0),
        tf.math.divide_no_nan(
            p[:self.num_thresholds - 1],
            tf.maximum(p[1:], 0),
            name='recall_relative_ratio'),
        tf.ones_like(p[1:]))

    pr_auc_increment = tf.math.divide_no_nan(
        prec_slope * (dtp + intercept * tf.math.log(safe_p_ratio)),
        tf.maximum(self.true_positives[1:] + self.false_negatives[1:], 0),
        name='pr_auc_increment')

    if self.multi_label:
      by_label_auc = tf.reduce_sum(
          pr_auc_increment, name=self.name + '_by_label', axis=0)

      if self._summarize:
        if self.label_weights is None:
          # Evenly weighted average of the label AUCs.
          return tf.reduce_mean(by_label_auc, name=self.name)
        else:
          # Weighted average of the label AUCs.
          return tf.math.divide_no_nan(
              tf.reduce_sum(
                  tf.multiply(by_label_auc, self.label_weights)),
              tf.reduce_sum(self.label_weights),
              name=self.name)
      else:
        return by_label_auc
    else:
      if self._summarize:
        return tf.reduce_sum(pr_auc_increment, name='interpolate_pr_auc')
      else:
        return pr_auc_increment


  def result(self):
    """Add option to remove summary.
    It's not clear why, but these metrics_utils == aren't working for tf.26 on.
    I'm hacking a solution to compare the values instead."""
    if (self.curve.value == metrics_utils.AUCCurve.PR.value and
        self.summation_method.value == metrics_utils.AUCSummationMethod.INTERPOLATION.value
       ):
      # This use case is different and is handled separately.
      return self.interpolate_pr_auc()

    # Set `x` and `y` values for the curves based on `curve` config.
    recall = tf.math.divide_no_nan(
        self.true_positives,
        tf.math.add(self.true_positives, self.false_negatives))
    if self.curve.value == metrics_utils.AUCCurve.ROC.value:
      fp_rate = tf.math.divide_no_nan(
          self.false_positives,
          tf.math.add(self.false_positives, self.true_negatives))
      x = fp_rate
      y = recall
    else:  # curve == 'PR'.
      precision = tf.math.divide_no_nan(
          self.true_positives,
          tf.math.add(self.true_positives, self.false_positives))
      x = recall
      y = precision

    # Find the rectangle heights based on `summation_method`.
    if self.summation_method.value == metrics_utils.AUCSummationMethod.INTERPOLATION.value:
      # Note: the case ('PR', 'interpolation') has been handled above.
      heights = (y[:self.num_thresholds - 1] + y[1:]) / 2.
    elif self.summation_method.value == metrics_utils.AUCSummationMethod.MINORING.value:
      heights = tf.minimum(y[:self.num_thresholds - 1], y[1:])
    else:  # self.summation_method = metrics_utils.AUCSummationMethod.MAJORING:
      heights = tf.maximum(y[:self.num_thresholds - 1], y[1:])

    # Sum up the areas of all the rectangles.
    if self.multi_label:
      riemann_terms = tf.multiply(x[:self.num_thresholds - 1] - x[1:],
                                        heights)
      by_label_auc = tf.reduce_sum(
          riemann_terms, name=self.name + '_by_label', axis=0)

      if self._summarize:
        if self.label_weights is None:
          # Unweighted average of the label AUCs.
          return tf.reduce_mean(by_label_auc, name=self.name)
        else:
          # Weighted average of the label AUCs.
          return tf.math.div_no_nan(
              tf.reduce_sum(
                  tf.multiply(by_label_auc, self.label_weights)),
              tf.reduce_sum(self.label_weights),
              name=self.name)
      else:
        return by_label_auc
    else:
      if self._summarize:
        return tf.reduce_sum(
            tf.multiply(x[:self.num_thresholds-1] - x[1:], heights),
            name=self.name)
      else:
        return tf.multiply(x[:self.num_thresholds-1] - x[1:], heights)


class PearsonR(tf.keras.metrics.Metric):
  def __init__(self, num_targets, summarize=True, name='pearsonr', **kwargs):
    super(PearsonR, self).__init__(name=name, **kwargs)
    self._summarize = summarize
    self._shape = (num_targets,)
    self._count = self.add_weight(name='count', shape=self._shape, initializer='zeros')

    self._product = self.add_weight(name='product', shape=self._shape, initializer='zeros')
    self._true_sum = self.add_weight(name='true_sum', shape=self._shape, initializer='zeros')
    self._true_sumsq = self.add_weight(name='true_sumsq', shape=self._shape, initializer='zeros')
    self._pred_sum = self.add_weight(name='pred_sum', shape=self._shape, initializer='zeros')
    self._pred_sumsq = self.add_weight(name='pred_sumsq', shape=self._shape, initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(y_pred, 'float32')

    if len(y_true.shape) == 2:
      reduce_axes = 0
    else:
      reduce_axes = [0,1]

    product = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=reduce_axes)
    self._product.assign_add(product)

    true_sum = tf.reduce_sum(y_true, axis=reduce_axes)
    self._true_sum.assign_add(true_sum)

    true_sumsq = tf.reduce_sum(tf.math.square(y_true), axis=reduce_axes)
    self._true_sumsq.assign_add(true_sumsq)

    pred_sum = tf.reduce_sum(y_pred, axis=reduce_axes)
    self._pred_sum.assign_add(pred_sum)

    pred_sumsq = tf.reduce_sum(tf.math.square(y_pred), axis=reduce_axes)
    self._pred_sumsq.assign_add(pred_sumsq)

    count = tf.ones_like(y_true)
    count = tf.reduce_sum(count, axis=reduce_axes)
    self._count.assign_add(count)

  def result(self):
    true_mean = tf.divide(self._true_sum, self._count)
    true_mean2 = tf.math.square(true_mean)
    pred_mean = tf.divide(self._pred_sum, self._count)
    pred_mean2 = tf.math.square(pred_mean)

    term1 = self._product
    term2 = -tf.multiply(true_mean, self._pred_sum)
    term3 = -tf.multiply(pred_mean, self._true_sum)
    term4 = tf.multiply(self._count, tf.multiply(true_mean, pred_mean))
    covariance = term1 + term2 + term3 + term4

    true_var = self._true_sumsq - tf.multiply(self._count, true_mean2)
    pred_var = self._pred_sumsq - tf.multiply(self._count, pred_mean2)
    pred_var = tf.where(tf.greater(pred_var, 1e-12),
                        pred_var,
                        np.inf*tf.ones_like(pred_var))
    
    tp_var = tf.multiply(tf.math.sqrt(true_var), tf.math.sqrt(pred_var))
    correlation = tf.divide(covariance, tp_var)

    if self._summarize:
        return tf.reduce_mean(correlation)
    else:
        return correlation

  def reset_state(self):
      K.batch_set_value([(v, np.zeros(self._shape)) for v in self.variables])


class R2(tf.keras.metrics.Metric):
  def __init__(self, num_targets, summarize=True, name='r2', **kwargs):
    super(R2, self).__init__(name=name, **kwargs)
    self._summarize = summarize
    self._shape = (num_targets,)
    self._count = self.add_weight(name='count', shape=self._shape, initializer='zeros')

    self._true_sum = self.add_weight(name='true_sum', shape=self._shape, initializer='zeros')
    self._true_sumsq = self.add_weight(name='true_sumsq', shape=self._shape, initializer='zeros')

    self._product = self.add_weight(name='product', shape=self._shape, initializer='zeros')
    self._pred_sumsq = self.add_weight(name='pred_sumsq', shape=self._shape, initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(y_pred, 'float32')

    if len(y_true.shape) == 2:
      reduce_axes = 0
    else:
      reduce_axes = [0,1]

    true_sum = tf.reduce_sum(y_true, axis=reduce_axes)
    self._true_sum.assign_add(true_sum)

    true_sumsq = tf.reduce_sum(tf.math.square(y_true), axis=reduce_axes)
    self._true_sumsq.assign_add(true_sumsq)

    product = tf.reduce_sum(tf.multiply(y_true, y_pred), axis=reduce_axes)
    self._product.assign_add(product)

    pred_sumsq = tf.reduce_sum(tf.math.square(y_pred), axis=reduce_axes)
    self._pred_sumsq.assign_add(pred_sumsq)

    count = tf.ones_like(y_true)
    count = tf.reduce_sum(count, axis=reduce_axes)
    self._count.assign_add(count)

  def result(self):
    true_mean = tf.divide(self._true_sum, self._count)
    true_mean2 = tf.math.square(true_mean)

    total = self._true_sumsq - tf.multiply(self._count, true_mean2)

    resid1 = self._pred_sumsq
    resid2 = -2*self._product
    resid3 = self._true_sumsq
    resid = resid1 + resid2 + resid3

    r2 = tf.ones_like(self._shape, dtype=tf.float32) - tf.divide(resid, total)

    if self._summarize:
        return tf.reduce_mean(r2)
    else:
        return r2

  def reset_state(self):
    K.batch_set_value([(v, np.zeros(self._shape)) for v in self.variables])
    
