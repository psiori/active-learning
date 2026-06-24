<template>
  <article
    class="thumb-card"
    :class="{
      'thumb-card--inactive': deselected,
      'thumb-card--loaded': thumbLoaded,
    }"
  >
    <button
      class="thumb-open-button"
      type="button"
      :aria-label="`Open ${item.sample_id}`"
      @click="emit('open', index)"
    >
      <div class="thumb-loading"></div>
      <img
        class="thumb-image"
        :class="{ 'thumb-image--loaded': thumbLoaded }"
        :src="thumbnailSrc"
        :alt="item.sample_id"
        loading="lazy"
        @load="thumbLoaded = true"
      />
      <img
        v-if="item.mask_url"
        class="mask-overlay"
        :src="maskSrc"
        :alt="`${item.sample_id} mask overlay`"
        :style="{ opacity: overlayOpacity }"
        loading="lazy"
      />
    </button>
    <button
      v-if="showSelectionState"
      class="thumb-selection-toggle"
      :class="{ 'thumb-selection-toggle--excluded': deselected }"
      type="button"
      :aria-label="deselected ? 'Excluded from export — click to include' : 'Included in export — click to exclude'"
      @click.stop="emit('toggle', item.sample_id)"
    >
      {{ deselected ? '✕' : '✓' }}
    </button>
  </article>
</template>

<script setup>
import { computed, ref, watch } from 'vue'

const props = defineProps({
  item: {
    type: Object,
    required: true,
  },
  index: {
    type: Number,
    required: true,
  },
  overlayOpacity: {
    type: Number,
    required: true,
  },
  deselected: {
    type: Boolean,
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
const thumbLoaded = ref(false)

const thumbnailSrc = computed(() => props.withCacheRoot(props.item.thumbnail_url))
const maskSrc = computed(() => (
  props.item.mask_url ? props.withCacheRoot(props.item.mask_url) : ''
))

watch(
  () => props.item.sample_id,
  () => {
    thumbLoaded.value = false
  },
)
</script>
