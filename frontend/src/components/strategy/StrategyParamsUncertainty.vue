<template>
  <div>
    <label class="field">
      <span>Uncertainty model</span>
      <select v-model="strategyParams.uncertainty_model" @change="emit('model-change')">
        <option v-for="model in modelOptions" :key="model.name" :value="model.name">
          {{ model.name }}
        </option>
      </select>
    </label>
    <label v-if="selectedUncertaintyModel?.url" class="field">
      <span>{{ selectedUncertaintyModel.name }} model URL</span>
      <input v-model="strategyParams.uncertainty_model_url" type="url" />
    </label>
    <div class="field">
      <span>Alpha</span>
      <RangeSlider
        :min="0"
        :max="1"
        :step="0.01"
        :min-value="strategyParams.alpha"
        :max-value="strategyParams.alpha"
        :single-value="true"
        :format-value="(v) => v.toFixed(2)"
        @update:max-value="strategyParams.alpha = $event"
      />
    </div>
    <label class="field">
      <span>Provider</span>
      <select v-model="strategyParams.provider">
        <option value="mc_dropout">mc_dropout</option>
        <option value="entropy">entropy</option>
        <option value="bald">bald</option>
      </select>
    </label>
    <label class="field">
      <span>MC iterations</span>
      <input v-model.number="strategyParams.mc_iterations" type="number" min="1" step="1" />
    </label>
    <label class="field">
      <span>Batch size</span>
      <input v-model.number="strategyParams.batch_size" type="number" min="1" step="1" />
    </label>
    <label class="field">
      <span>Aggregation</span>
      <select v-model="strategyParams.aggregation">
        <option value="mean">mean</option>
        <option value="topk_mean">topk_mean</option>
        <option value="max">max</option>
      </select>
    </label>
    <div class="field">
      <span>Top-k fraction</span>
      <RangeSlider
        :min="0.01"
        :max="1"
        :step="0.01"
        :min-value="strategyParams.topk_fraction"
        :max-value="strategyParams.topk_fraction"
        :single-value="true"
        :format-value="(v) => v.toFixed(2)"
        @update:max-value="strategyParams.topk_fraction = $event"
      />
    </div>
    <div class="field">
      <span>Candidate multiplier</span>
      <RangeSlider
        :min="2"
        :max="20"
        :step="1"
        :min-value="strategyParams.candidate_multiplier"
        :max-value="strategyParams.candidate_multiplier"
        :single-value="true"
        @update:max-value="strategyParams.candidate_multiplier = $event"
      />
    </div>
  </div>
</template>

<script setup>
import RangeSlider from '@/components/base/RangeSlider.vue'

defineProps({
  strategyParams: {
    type: Object,
    required: true,
  },
  modelOptions: {
    type: Array,
    required: true,
  },
  selectedUncertaintyModel: {
    type: Object,
    default: null,
  },
})

const emit = defineEmits(['model-change'])
</script>
