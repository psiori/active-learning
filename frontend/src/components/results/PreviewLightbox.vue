<template>
  <Teleport to="body">
    <div class="lightbox-backdrop" role="presentation" @click="emit('close')">
      <section
        class="lightbox-panel"
        role="dialog"
        aria-modal="true"
        aria-label="Image preview"
        @click.stop
      >
        <header class="lightbox-header">
          <div class="lightbox-header-main">
            <label v-if="showOverlayControls" class="field lightbox-slider">
              <span>Mask overlay strength</span>
              <input :value="overlayAlpha" type="range" min="0" max="100" step="1" @input="handleSliderInput" />
            </label>
            <button
              v-if="showSelectionState"
              class="lightbox-exclusion-toggle"
              :class="{ 'is-excluded': isExcluded }"
              type="button"
              :aria-label="isExcluded ? 'Excluded from export — click to include' : 'Included in export — click to exclude'"
              @click.stop="emit('toggle')"
            >
              {{ isExcluded ? '✕ Excluded' : '✓ Included' }}
            </button>
          </div>
          <button class="lightbox-close" type="button" aria-label="Close preview" @click="emit('close')">
            ×
          </button>
        </header>

        <div class="lightbox-stage">
          <button
            class="lightbox-nav lightbox-nav--prev"
            type="button"
            aria-label="Previous image"
            :disabled="!hasMultipleItems"
            @click="emit('previous')"
          >
            ‹
          </button>

          <figure ref="figureRef" class="lightbox-figure">
            <div class="lightbox-media" :style="mediaBoxStyle">
              <Transition name="lightbox-swap">
                <div :key="item.sample_id" class="lightbox-frame">
                  <div v-if="!imageLoaded" class="lightbox-loading"></div>
                  <img
                    class="lightbox-image"
                    :class="{ 'lightbox-image--loaded': imageLoaded }"
                    :src="imageSrc"
                    :alt="item.sample_id"
                    @load="handleImageLoad"
                  />
                  <Transition name="lightbox-overlay-fade">
                    <img
                      v-if="maskSrc && imageLoaded"
                      class="lightbox-overlay"
                      :src="maskSrc"
                      :alt="`${item.sample_id} mask overlay`"
                      :style="{ '--lightbox-overlay-opacity': overlayOpacity }"
                    />
                  </Transition>
                </div>
              </Transition>
            </div>
          </figure>

          <button
            class="lightbox-nav lightbox-nav--next"
            type="button"
            aria-label="Next image"
            :disabled="!hasMultipleItems"
            @click="emit('next')"
          >
            ›
          </button>
        </div>

        <footer class="lightbox-footer">
          <span><strong>ID</strong> {{ item.sample_id }}</span>
          <span><strong>Resolution</strong> {{ resolutionText }}</span>
          <span><strong>Date</strong> {{ dateText }}</span>
          <span><strong>Time</strong> {{ timeText }}</span>
        </footer>
      </section>
    </div>
  </Teleport>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref, watch, watchEffect } from 'vue'
import { parseTimestampFromSampleId } from '@/utils/sampleId'

const props = defineProps({
  item: {
    type: Object,
    required: true,
  },
  previousItem: {
    type: Object,
    default: null,
  },
  nextItem: {
    type: Object,
    default: null,
  },
  overlayAlpha: {
    type: Number,
    required: true,
  },
  overlayOpacity: {
    type: Number,
    required: true,
  },
  showOverlayControls: {
    type: Boolean,
    required: true,
  },
  showSelectionState: {
    type: Boolean,
    required: true,
  },
  isExcluded: {
    type: Boolean,
    required: true,
  },
  withCacheRoot: {
    type: Function,
    required: true,
  },
  hasMultipleItems: {
    type: Boolean,
    required: true,
  },
})

const emit = defineEmits(['close', 'next', 'previous', 'update:overlayAlpha', 'toggle'])
const naturalSize = ref({ width: 0, height: 0 })
const boxSize = ref({ width: 0, height: 0 })
const imageLoaded = ref(false)
const prefetchedUrls = new Set()
const prefetchedSizes = new Map()
const figureRef = ref(null)
let resizeObserver = null

const imageSrc = computed(() => withFullResolution(props.withCacheRoot(props.item.thumbnail_url)))
const maskSrc = computed(() => (
  props.item.mask_url ? props.withCacheRoot(props.item.mask_url) : ''
))
const parsedTimestamp = computed(() => parseTimestampFromSampleId(props.item.sample_id))
const dateText = computed(() => parsedTimestamp.value?.date ?? 'Unknown')
const timeText = computed(() => parsedTimestamp.value?.time ?? 'Unknown')
const resolutionText = computed(() => {
  if (naturalSize.value.width > 0 && naturalSize.value.height > 0) {
    return `${naturalSize.value.width} × ${naturalSize.value.height}`
  }
  return 'Unknown'
})
const activeImageSize = computed(() => {
  if (naturalSize.value.width > 0 && naturalSize.value.height > 0) {
    return naturalSize.value
  }
  return prefetchedSizes.get(imageSrc.value) || { width: 0, height: 0 }
})
const mediaBoxStyle = computed(() => {
  if (boxSize.value.width > 0 && boxSize.value.height > 0) {
    return {
      width: `${boxSize.value.width}px`,
      height: `${boxSize.value.height}px`,
    }
  }
  return {}
})

watchEffect(() => {
  prefetchImage(imageSrc.value)

  if (props.hasMultipleItems) {
    if (props.previousItem?.thumbnail_url) {
      prefetchImage(withFullResolution(props.withCacheRoot(props.previousItem.thumbnail_url)))
    }
    if (props.nextItem?.thumbnail_url) {
      prefetchImage(withFullResolution(props.withCacheRoot(props.nextItem.thumbnail_url)))
    }
  }
})

watchEffect(() => {
  const width = activeImageSize.value.width
  const height = activeImageSize.value.height
  const figure = figureRef.value
  if (!figure || width <= 0 || height <= 0) {
    return
  }
  updateBoxSize(figure.clientWidth, availableMediaHeight(), width, height)
})

watch(
  () => props.item.sample_id,
  () => {
    naturalSize.value = { width: 0, height: 0 }
    boxSize.value = { width: 0, height: 0 }
    imageLoaded.value = false
  },
)

onMounted(() => {
  if (!window.ResizeObserver || !figureRef.value) {
    return
  }

  resizeObserver = new window.ResizeObserver((entries) => {
    const entry = entries[0]
    if (!entry) {
      return
    }
    const width = activeImageSize.value.width
    const height = activeImageSize.value.height
    if (width <= 0 || height <= 0) {
      return
    }
    updateBoxSize(entry.contentRect.width, availableMediaHeight(), width, height)
  })
  resizeObserver.observe(figureRef.value)
})

onUnmounted(() => {
  resizeObserver?.disconnect()
})

function handleSliderInput(event) {
  emit('update:overlayAlpha', Number(event.target.value))
}

function handleImageLoad(event) {
  imageLoaded.value = true
  naturalSize.value = {
    width: event.target.naturalWidth,
    height: event.target.naturalHeight,
  }
}

function withFullResolution(url) {
  const separator = url.includes('?') ? '&' : '?'
  return `${url}${separator}use_full_res_images=true`
}

function prefetchImage(url) {
  if (!url || prefetchedUrls.has(url)) {
    return
  }

  prefetchedUrls.add(url)
  const image = new Image()
  image.onload = () => {
    prefetchedSizes.set(url, {
      width: image.naturalWidth,
      height: image.naturalHeight,
    })
    if (url === imageSrc.value && !imageLoaded.value && figureRef.value) {
      updateBoxSize(
        figureRef.value.clientWidth,
        availableMediaHeight(),
        image.naturalWidth,
        image.naturalHeight,
      )
    }
  }
  image.src = url
}

function availableMediaHeight() {
  if (typeof window === 'undefined') {
    return 900
  }
  return Math.min(window.innerHeight * 0.68, 900)
}

function updateBoxSize(availableWidth, availableHeight, imageWidth, imageHeight) {
  if (availableWidth <= 0 || availableHeight <= 0 || imageWidth <= 0 || imageHeight <= 0) {
    return
  }

  const imageRatio = imageWidth / imageHeight
  const boundsRatio = availableWidth / availableHeight
  if (boundsRatio > imageRatio) {
    boxSize.value = {
      width: availableHeight * imageRatio,
      height: availableHeight,
    }
    return
  }

  boxSize.value = {
    width: availableWidth,
    height: availableWidth / imageRatio,
  }
}

</script>
