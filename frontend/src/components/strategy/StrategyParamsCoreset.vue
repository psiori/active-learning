<template>
  <label class="field">
    <span>Feature model</span>
    <select v-model="strategyParams.feature_model">
      <option v-for="model in modelOptions" :key="model.name" :value="model.name">
        {{ model.name }}
      </option>
    </select>
  </label>
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
})
</script>
