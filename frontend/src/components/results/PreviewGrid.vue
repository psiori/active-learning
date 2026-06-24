<template>
  <section class="grid-panel">
    <PreviewGridItem
      v-for="(item, index) in previewItems"
      :key="item.sample_id"
      :item="item"
      :index="index"
      :overlay-opacity="overlayOpacity"
      :deselected="isDeselected(item.sample_id)"
      :show-selection-state="showSelectionState"
      :with-cache-root="withCacheRoot"
      @open="emit('open', $event)"
      @toggle="emit('toggle', $event)"
    />
  </section>
</template>

<script setup>
import PreviewGridItem from './PreviewGridItem.vue'

defineProps({
  previewItems: {
    type: Array,
    required: true,
  },
  overlayOpacity: {
    type: Number,
    required: true,
  },
  isDeselected: {
    type: Function,
    required: true,
  },
  showSelectionState: {
    type: Boolean,
    required: true,
  },
  withCacheRoot: {
    type: Function,
    required: true,
  },
})

const emit = defineEmits(['open', 'toggle'])
</script>
