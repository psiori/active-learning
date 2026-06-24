<template>
  <div>
    <label class="field">
      <span>ALGES model</span>
      <select v-model="strategyParams.alges_model" @change="emit('model-change')">
        <option v-for="model in modelOptions" :key="model.name" :value="model.name">
          {{ model.name }}
        </option>
      </select>
    </label>
    <label v-if="selectedAlgesModel?.url" class="field">
      <span>{{ selectedAlgesModel.name }} model URL</span>
      <input v-model="strategyParams.alges_model_url" type="url" />
    </label>
    <label class="field">
      <span>Method</span>
      <select v-model="strategyParams.method">
        <option value="image">image</option>
        <option value="semantic">semantic</option>
      </select>
    </label>
    <label class="field">
      <span>ALGES batch size</span>
      <input v-model.number="strategyParams.alges_batch_size" type="number" min="1" step="1" />
    </label>
  </div>
</template>

<script setup>
defineProps({
  strategyParams: {
    type: Object,
    required: true,
  },
  modelOptions: {
    type: Array,
    required: true,
  },
  selectedAlgesModel: {
    type: Object,
    default: null,
  },
})

const emit = defineEmits(['model-change'])
</script>
