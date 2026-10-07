import { useEffect, useState } from "react";

// Runs `loader` whenever `deps` change and tracks loading / error / data state.
// Ignores responses from requests that were superseded by a newer one.
export function useFetch(loader, deps) {
  const [state, setState] = useState({ data: null, error: null, loading: true });

  useEffect(() => {
    let active = true;
    setState((s) => ({ ...s, loading: true, error: null }));
    loader()
      .then((data) => active && setState({ data, error: null, loading: false }))
      .catch((error) => active && setState({ data: null, error: error.message, loading: false }));
    return () => {
      active = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return state;
}
