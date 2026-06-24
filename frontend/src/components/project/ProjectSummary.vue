<template>
  <div v-if="projectConfig">
    <label class="field">
      <span>Query sensor</span>
      <input :value="projectConfig.project.query_sensor || 'n/a'" type="text" readonly disabled />
    </label>
    <dl class="summary-grid">
      <div>
        <dt>Seeded in Sama project</dt>
        <dd>{{ seededIdsCount ?? 'n/a' }}</dd>
      </div>
    </dl>
  </div>
</template>

<script setup>
defineProps({
  projectConfig: {
    type: Object,
    default: null,
  },
  seededIdsCount: {
    type: Number,
    default: null,
  },
})
</script>
