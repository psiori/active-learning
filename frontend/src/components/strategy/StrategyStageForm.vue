<template>
  <form class="query-form" @submit.prevent="emit('submit')">
    <label class="field">
      <span>Selection strategy</span>
      <select v-model="selectionForm.strategy" :disabled="disabledSelect">
        <option v-for="strategy in availableStrategies" :key="strategy" :value="strategy">
          {{ strategy }}
        </option>
      </select>
    </label>

    <label class="field">
      <span>Number to select</span>
      <input v-model.number="selectionForm.n_select" type="number" min="1" step="1" />
    </label>

    <StrategyParamsCoreset
      v-if="usesCoresetParams"
      :strategy-params="strategyParams"
      :model-options="featureModels"
    />

    <StrategyParamsUncertainty
      v-if="usesUncertaintyParams"
      :strategy-params="strategyParams"
      :model-options="uncertaintyModels"
      :selected-uncertainty-model="selectedUncertaintyModel"
      @model-change="emit('sync-model-urls')"
    />

    <StrategyParamsAlges
      v-if="usesAlgesParams"
      :strategy-params="strategyParams"
      :model-options="uncertaintyModels"
      :selected-alges-model="selectedAlgesModel"
      @model-change="emit('sync-model-urls')"
    />

    <button
      v-if="cancellingStrategy"
      class="primary-button"
      type="button"
      disabled
    >
      Cancelling…
    </button>
    <button
      v-else-if="loadingStrategy"
      class="primary-button"
      type="button"
      @click="emit('cancel')"
    >
      Cancel selection
    </button>
    <button v-else class="primary-button" type="submit" :disabled="disabledSubmit">
      Run selection
    </button>
  </form>
</template>

<script setup>
import StrategyParamsAlges from './StrategyParamsAlges.vue'
import StrategyParamsCoreset from './StrategyParamsCoreset.vue'
import StrategyParamsUncertainty from './StrategyParamsUncertainty.vue'

defineProps({
  selectionForm: {
    type: Object,
    required: true,
  },
  strategyParams: {
    type: Object,
    required: true,
  },
  availableStrategies: {
    type: Array,
    required: true,
  },
  featureModels: {
    type: Array,
    required: true,
  },
  uncertaintyModels: {
    type: Array,
    required: true,
  },
  usesCoresetParams: {
    type: Boolean,
    required: true,
  },
  usesUncertaintyParams: {
    type: Boolean,
    required: true,
  },
  usesAlgesParams: {
    type: Boolean,
    required: true,
  },
  selectedUncertaintyModel: {
    type: Object,
    default: null,
  },
  selectedAlgesModel: {
    type: Object,
    default: null,
  },
  loadingStrategy: {
    type: Boolean,
    required: true,
  },
  disabledSelect: {
    type: Boolean,
    required: true,
  },
  disabledSubmit: {
    type: Boolean,
    required: true,
  },
  cancellingStrategy: {
    type: Boolean,
    default: false,
  },
})

const emit = defineEmits(['submit', 'cancel', 'sync-model-urls'])
</script>
