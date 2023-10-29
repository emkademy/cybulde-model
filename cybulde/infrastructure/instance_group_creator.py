import time

from google.cloud import compute_v1

from cybulde.infrastructure.instance_template_creator import InstanceTemplateCreator
from cybulde.utils.gcp_utils import wait_for_extended_operation
from cybulde.utils.utils import get_logger


class InstanceGroupCreator:
    def __init__(
        self,
        instance_template_creator: InstanceTemplateCreator,
        name: str,
        node_count: int,
        project_id: str,
        zone: str,
    ):
        self.logger = get_logger(self.__class__.__name__)
        self.instance_template_creator = instance_template_creator
        self.name = name.lower()
        self.node_count = node_count
        self.project_id = project_id
        self.zone = zone

    def launch_instance_group(self) -> list[int]:
        instance_group = self._create_instance_group()
        self.logger.debug(f"{instance_group=}")

        instance_ids = self._get_instance_ids(self.name, self.node_count)
        return instance_ids

    def _create_instance_group(self) -> compute_v1.InstanceGroupManager:
        self.logger.info("Starting to create instance group...")
        instance_template = self.instance_template_creator.create_template()

        name = self.name
        instance_group_manager_resource = compute_v1.InstanceGroupManager(
            name=name,
            base_instance_name=self.name,
            instance_template=instance_template.self_link,
            target_size=self.node_count,
        )

        project_id = self.project_id
        zone = self.zone

        instance_group_managers_client = compute_v1.InstanceGroupManagersClient()
        operation = instance_group_managers_client.insert(
            project=project_id, instance_group_manager_resource=instance_group_manager_resource, zone=zone
        )
        wait_for_extended_operation(operation, "managed instance group creation")

        self.logger.info("Instance group has been created...")
        return instance_group_managers_client.get(project=project_id, instance_group_manager=name, zone=zone)

    def _get_instance_ids(self, name: str, node_count: int) -> list[int]:
        instance_ids = set()
        trial = 0
        max_trials = 10
        base_sleep_time = 1.5
        while trial <= max_trials:
            self.logger.info(f"Waiting for instances ({trial=})...")
            pager = self.list_instances_in_group(name)
            for instance in pager:
                if instance.id:
                    self.logger.info(f"Instance {instance.id} ready")
                    instance_ids.add(instance.id)

            if len(instance_ids) >= node_count:
                break

            time.sleep(pow(base_sleep_time, trial))
            trial += 1
        return list(instance_ids)

    def list_instances_in_group(
        self, name: str
    ) -> compute_v1.services.instance_group_managers.pagers.ListManagedInstancesPager:
        instance_group_managers_client = compute_v1.InstanceGroupManagersClient()
        pager = instance_group_managers_client.list_managed_instances(
            project=self.project_id, instance_group_manager=name, zone=self.zone
        )
        return pager
