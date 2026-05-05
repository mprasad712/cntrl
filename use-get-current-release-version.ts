import useAuthStore from "@/stores/authStore";
import { useDarkStore } from "@/stores/darkStore";
import { api } from "../../api";
import { getURL } from "../../helpers/constants";
import { UseRequestProcessor } from "../../services/request-processor";

export const useGetCurrentReleaseVersionQuery = (options?: any) => {
  const { query } = UseRequestProcessor();
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated);

  const responseFn = async (): Promise<string> => {
    const res = await api.get(`${getURL("RELEASES")}/current-version`);
    const version: string = res.data?.version ?? "";
    if (version) {
      useDarkStore.getState().refreshCurrentReleaseVersion(version);
    }
    return version;
  };

  return query(["useGetCurrentReleaseVersionQuery"], responseFn, {
    refetchOnWindowFocus: false,
    ...options,
    enabled: isAuthenticated && (options?.enabled ?? true),
  });
};
