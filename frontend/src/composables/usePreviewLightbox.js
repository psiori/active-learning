import { computed, ref, watch } from 'vue'
import { useEventListener, useScrollLock } from '@vueuse/core'

export function usePreviewLightbox(items) {
  const lightboxIndex = ref(null)
  const bodyScrollLock = useScrollLock(document.body)

  const lightboxItem = computed(() => {
    if (lightboxIndex.value == null) {
      return null
    }
    return items.value[lightboxIndex.value] || null
  })

  const lightboxPreviousItem = computed(() => {
    if (!items.value.length || lightboxIndex.value == null) {
      return null
    }
    return items.value[(lightboxIndex.value - 1 + items.value.length) % items.value.length] || null
  })

  const lightboxNextItem = computed(() => {
    if (!items.value.length || lightboxIndex.value == null) {
      return null
    }
    return items.value[(lightboxIndex.value + 1) % items.value.length] || null
  })

  watch(items, (nextItems) => {
    if (!nextItems.length) {
      closeLightbox()
      return
    }

    if (lightboxIndex.value != null && lightboxIndex.value >= nextItems.length) {
      lightboxIndex.value = nextItems.length - 1
    }
  })

  watch(lightboxIndex, (index) => {
    bodyScrollLock.value = index != null
  })

  useEventListener(window, 'keydown', (event) => {
    if (lightboxIndex.value == null) {
      return
    }

    if (event.key === 'Escape') {
      closeLightbox()
    } else if (event.key === 'ArrowLeft') {
      showPreviousLightboxItem()
    } else if (event.key === 'ArrowRight') {
      showNextLightboxItem()
    }
  })

  function openLightbox(index) {
    lightboxIndex.value = index
  }

  function closeLightbox() {
    lightboxIndex.value = null
  }

  function showPreviousLightboxItem() {
    if (!items.value.length || lightboxIndex.value == null) {
      return
    }

    lightboxIndex.value = (lightboxIndex.value - 1 + items.value.length) % items.value.length
  }

  function showNextLightboxItem() {
    if (!items.value.length || lightboxIndex.value == null) {
      return
    }

    lightboxIndex.value = (lightboxIndex.value + 1) % items.value.length
  }

  return {
    lightboxItem,
    lightboxPreviousItem,
    lightboxNextItem,
    openLightbox,
    closeLightbox,
    showPreviousLightboxItem,
    showNextLightboxItem,
  }
}
