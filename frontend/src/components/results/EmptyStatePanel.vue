<template>
  <section class="empty-panel">
    <template v-if="exportResult">
      <div>
        <p>{{ exportResult.message }}</p>
        <p v-if="exportResult.description">{{ exportResult.description }}</p>
      </div>
    </template>
    <p v-else>{{ emptyStateText }}</p>
  </section>
</template>

<script setup>
defineProps({
  exportResult: {
    type: Object,
    default: null,
  },
  emptyStateText: {
    type: String,
    required: true,
  },
})
</script>
