import { computed, nextTick, ref, watch } from 'vue'
import { useEventListener, useMediaQuery } from '@vueuse/core'

export function usePreviewPaneOffset({
  elementRef,
  showPreviewGrid,
}) {
  const offsetPx = ref(0)
  const isMobile = useMediaQuery('(max-width: 767px)')
  const spacerStyle = computed(() => ({
    height: `${offsetPx.value}px`,
  }))
  let lastScrollY = 0

  watch(showPreviewGrid, async (isVisible, wasVisible) => {
    if (typeof window === 'undefined') {
      return
    }

    lastScrollY = window.scrollY

    if (!isVisible || isMobile.value) {
      offsetPx.value = 0
      return
    }

    if (wasVisible) {
      return
    }

    await nextTick()

    const element = elementRef.value
    if (!element) {
      return
    }

    const { top } = element.getBoundingClientRect()
    offsetPx.value = top < 0 ? -top : 0
  })

  useEventListener(window, 'scroll', () => {
    if (typeof window === 'undefined' || isMobile.value || offsetPx.value <= 0) {
      if (typeof window !== 'undefined') {
        lastScrollY = window.scrollY
      }
      return
    }

    const deltaY = window.scrollY - lastScrollY
    lastScrollY = window.scrollY

    if (deltaY >= 0) {
      return
    }

    const element = elementRef.value
    if (!element) {
      return
    }

    const { top } = element.getBoundingClientRect()
    if (top < 0) {
      return
    }

    offsetPx.value = Math.max(0, offsetPx.value + deltaY)
  }, { passive: true })

  return {
    offsetPx,
    spacerStyle,
  }
}
