<template>
  <section v-if="exportResult" class="stats-panel">
    <div class="stat-card">
      <span>Action</span>
      <strong>{{ exportActionLabel }}</strong>
    </div>
    <div v-if="exportResult.updated_count != null" class="stat-card">
      <span>Updated</span>
      <strong>{{ exportResult.updated_count }}</strong>
    </div>
    <div v-else-if="exportResult.image_count != null" class="stat-card">
      <span>Images</span>
      <strong>{{ exportResult.image_count }}</strong>
    </div>
    <div v-if="exportResult.export_id" class="stat-card">
      <span>Export ID</span>
      <strong>{{ exportResult.export_id }}</strong>
    </div>
    <div v-if="exportResult.sama_batch_id" class="stat-card">
      <span>Sama batch</span>
      <strong>{{ exportResult.sama_batch_id }}</strong>
    </div>
  </section>

  <section v-else-if="activeResult" class="stats-panel">
    <div class="stat-card">
      <span>Labeled</span>
      <strong>{{ activeResult.labeled_ids_count }} of {{ activeResult.all_ids_count }}</strong>
    </div>
    <div v-if="activeResult.brightness_filtered_ids_count < activeResult.pool_ids_count" class="stat-card">
      <span>Brightness filtered</span>
      <strong>{{ activeResult.pool_ids_count - activeResult.brightness_filtered_ids_count }} of {{ activeResult.pool_ids_count }}</strong>
    </div>
    <div class="stat-card">
      <span>{{ activeResult.brightness_filtered_ids_count < activeResult.pool_ids_count ? 'Remaining pool' : 'Pool' }}</span>
      <strong>{{ activeResult.brightness_filtered_ids_count }}</strong>
    </div>
    <div v-if="activeResult.result_kind === 'strategy'" class="stat-card">
      <span>Selected</span>
      <strong>{{ selectedStrategyIds.length }} of {{ activeResult.selected_ids_count }}</strong>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  exportResult: {
    type: Object,
    default: null,
  },
  activeResult: {
    type: Object,
    default: null,
  },
  selectedStrategyIds: {
    type: Array,
    required: true,
  },
})

const exportActionLabel = computed(() => {
  const k = props.exportResult?.kind
  if (k === 'overlay') return 'Overlay mosaic'
  if (k === 'exclusion_tags') return 'Exclusion tags (CRID)'
  return 'Seed to Sama'
})
</script>
