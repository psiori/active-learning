<template>
  <PreviewGrid
    :preview-items="previewItems"
    :overlay-opacity="overlayOpacity"
    :is-deselected="isDeselected"
    :show-selection-state="showSelectionState"
    :with-cache-root="withCacheRoot"
    @open="openLightbox"
    @toggle="emit('toggle', $event)"
  />

  <PreviewLightbox
    v-if="lightboxItem"
    :item="lightboxItem"
    :previous-item="lightboxPreviousItem"
    :next-item="lightboxNextItem"
    :overlay-alpha="overlayAlpha"
    :overlay-opacity="overlayOpacity"
    :show-overlay-controls="showOverlayControls"
    :with-cache-root="withCacheRoot"
    :has-multiple-items="previewItems.length > 1"
    @close="closeLightbox"
    @next="showNextLightboxItem"
    @previous="showPreviousLightboxItem"
    @update:overlay-alpha="emit('update:overlayAlpha', $event)"
  />
</template>

<script setup>
import { toRef } from 'vue'
import { usePreviewLightbox } from '@/composables/usePreviewLightbox'
import PreviewGrid from './PreviewGrid.vue'
import PreviewLightbox from './PreviewLightbox.vue'

const props = defineProps({
  previewItems: {
    type: Array,
    required: true,
  },
  overlayAlpha: {
    type: Number,
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
  showOverlayControls: {
    type: Boolean,
    required: true,
  },
  withCacheRoot: {
    type: Function,
    required: true,
  },
})

const emit = defineEmits(['toggle', 'update:overlayAlpha'])

const {
  lightboxItem,
  lightboxPreviousItem,
  lightboxNextItem,
  openLightbox,
  closeLightbox,
  showPreviousLightboxItem,
  showNextLightboxItem,
} = usePreviewLightbox(toRef(props, 'previewItems'))
</script>
