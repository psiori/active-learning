<template>
  <header class="results-header">
    <div>
      <p class="stage-kicker">Preview</p>
      <h2 v-if="resultTitle">{{ resultTitle }}</h2>
    </div>
    <p v-if="errorMessage" class="error-text">{{ errorMessage }}</p>
  </header>
</template>

<script setup>
defineProps({
  resultTitle: {
    type: String,
    default: '',
  },
  errorMessage: {
    type: String,
    default: '',
  },
})
</script>
